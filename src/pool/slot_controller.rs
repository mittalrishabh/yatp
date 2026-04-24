// Copyright 2024 TiKV Project Authors. Licensed under Apache-2.0.

use parking_lot_core::{ParkToken, UnparkToken};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// RAII guard that releases a slot on drop.
pub struct SlotGuard {
    controller: Arc<SlotController>,
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        self.controller.release();
    }
}

/// CockroachDB-style CPU-based slot controller.
/// Limits concurrent task execution based on available CPU capacity.
/// `max_slots` is adjusted externally (e.g. by a resource quota worker).
/// When over capacity, worker threads park and are woken by `release()`.
pub struct SlotController {
    max_slots: AtomicUsize,
    in_flight: AtomicUsize,
    enabled: AtomicBool,
}

impl SlotController {
    pub fn new(initial_slots: usize) -> Arc<Self> {
        Arc::new(SlotController {
            max_slots: AtomicUsize::new(initial_slots),
            in_flight: AtomicUsize::new(0),
            enabled: AtomicBool::new(true),
        })
    }

    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    pub fn set_max_slots(&self, slots: usize) {
        self.max_slots.store(slots, Ordering::Relaxed);
    }

    pub fn max_slots(&self) -> usize {
        self.max_slots.load(Ordering::Relaxed)
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }

    /// Acquires a slot and returns a guard that releases it on drop.
    pub fn acquire_guard(self: &Arc<Self>) -> SlotGuard {
        self.acquire();
        SlotGuard {
            controller: self.clone(),
        }
    }

    /// Acquires a slot. Parks the calling thread if over capacity until a slot
    /// is available. Always succeeds — caller must call `release()` after.
    pub fn acquire(&self) {
        if !self.enabled.load(Ordering::Relaxed) {
            self.in_flight.fetch_add(1, Ordering::Relaxed);
            return;
        }
        loop {
            let max = self.max_slots.load(Ordering::Relaxed);
            let prev = self.in_flight.fetch_add(1, Ordering::Relaxed);
            if prev < max {
                return;
            }
            // Over capacity — undo increment and park until release() wakes us.
            self.in_flight.fetch_sub(1, Ordering::Relaxed);
            let addr = self as *const SlotController as usize;
            unsafe {
                parking_lot_core::park(
                    addr,
                    || {
                        // Recheck under park lock to avoid lost wakeup.
                        let max = self.max_slots.load(Ordering::Relaxed);
                        let in_flight = self.in_flight.load(Ordering::Relaxed);
                        in_flight >= max
                    },
                    || {},
                    |_, _| {},
                    ParkToken(0),
                    None,
                );
            }
        }
    }

    /// Releases a slot and wakes one parked thread if below capacity.
    pub fn release(&self) {
        let prev = self.in_flight.fetch_sub(1, Ordering::Relaxed);
        let max = self.max_slots.load(Ordering::Relaxed);
        if prev <= max {
            let addr = self as *const SlotController as usize;
            unsafe {
                parking_lot_core::unpark_one(addr, |_| UnparkToken(0));
            }
        }
    }
}
