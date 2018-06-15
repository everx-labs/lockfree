use std::{
    cell::RefCell,
    collections::VecDeque,
    mem::transmute,
    ptr::NonNull,
    sync::atomic::{AtomicUsize, Ordering::*},
};

/// Adds the given pointer and drop function to the local deletion queue.
/// If there is no critical code executing (i.e. the incinerator is not
/// paused), all local queue items are deleted. This function is unsafe because
/// pointers must be correctly dropped such as no "use after free" or "double
/// free" happens.
pub unsafe fn add<T>(ptr: NonNull<T>, dropper: fn(NonNull<T>)) {
    LOCAL_DELETION.with(|queue| {
        // First of all, let's put it on the queue because of a possible
        // obstruction when deleting.
        queue.add(Garbage {
            ptr: NonNull::new_unchecked(ptr.as_ptr() as *mut u8),
            dropper: transmute(dropper),
        });
        if PAUSED_COUNT.load(SeqCst) == 0 {
            // Please, note that we check for the counter AFTER the enqueueing.
            // This ensures that no pointer is added after a possible status
            // change. All pointers deleted here were already added
            // to the queue.
            queue.delete();
        }
    })
}

/// Tries to force deletion of all local queue items. Only succeeds
/// if there are no pauses when checking for them before the deletion.
/// Please note this functions is not strictly need to be called, but
/// it may help on releasing garbage if you added a lot of them during
/// a pause.
pub fn try_force() -> Result<(), ()> {
    LOCAL_DELETION.with(|queue| {
        if PAUSED_COUNT.load(SeqCst) == 0 {
            // No problem to change the status while deleting.
            // No pointer is added to the queue during the change.
            queue.delete();
            Ok(())
        } else {
            Err(())
        }
    })
}

/// Pauses the incinerator and executes the given function as critical code.
/// No deletions of new queues will start during the execution of the given
/// function.
pub fn pause<F, T>(exec: F) -> T
where
    F: FnOnce() -> T,
{
    // Do not allow deletions, but allow adding pointers to the local queues.
    PAUSED_COUNT.fetch_add(1, SeqCst);
    let res = exec();
    // After the execution, everything is fine.
    PAUSED_COUNT.fetch_sub(1, SeqCst);
    res
}

struct Garbage {
    ptr: NonNull<u8>,
    dropper: fn(NonNull<u8>),
}

struct GarbageQueue {
    inner: RefCell<VecDeque<Garbage>>,
}

impl GarbageQueue {
    fn new() -> Self {
        Self {
            inner: RefCell::new(VecDeque::with_capacity(16)),
        }
    }

    fn add(&self, garbage: Garbage) {
        self.inner.borrow_mut().push_back(garbage);
    }

    fn delete(&self) {
        let mut deque = self.inner.borrow_mut();
        while let Some(garbage) = deque.pop_front() {
            (garbage.dropper)(garbage.ptr);
        }
    }
}

impl Drop for GarbageQueue {
    fn drop(&mut self) {
        while PAUSED_COUNT.load(SeqCst) != 0 {}
        self.delete();
    }
}

thread_local! {
    static LOCAL_DELETION: GarbageQueue = GarbageQueue::new();
}

static PAUSED_COUNT: AtomicUsize = AtomicUsize::new(0);