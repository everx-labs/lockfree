use core::{
    cell::UnsafeCell,
    mem::{size_of, MaybeUninit},
    ops::{Deref, DerefMut},
    sync::atomic::{fence, AtomicBool, AtomicUsize, Ordering::*},
};

#[cfg(feature = "std")]
use std::sync::Arc;

const USIZE_BITS: usize = size_of::<usize>() * 8;
const USIZE_HIGHEST_BIT_MASK: usize = 1 << (USIZE_BITS - 1);
const USIZE_LOWER_BITS_MASK: usize = (1 << (USIZE_BITS - 1)) - 1;

const fn data_bits(bits: usize) -> usize {
    bits & USIZE_LOWER_BITS_MASK
}

const fn epoch_bit(bits: usize) -> usize {
    bits >> (USIZE_BITS - 1)
}

const fn set_data_bits(bits: usize, data_bits: usize) -> usize {
    (bits & USIZE_HIGHEST_BIT_MASK) | (data_bits & USIZE_LOWER_BITS_MASK)
}

const fn set_epoch_bit(bits: usize, epoch_bit: usize) -> usize {
    (bits & USIZE_LOWER_BITS_MASK) | (epoch_bit << (USIZE_BITS - 1))
}

const NULL_IDX: usize = !0;

/// Initializes a slab array.
macro_rules! make_slab_array {
    ($size:expr) => {
        unsafe {
            use core::mem::{transmute, MaybeUninit};

            let mut arr = MaybeUninit::<[MaybeUninit<_>; $size]>::uninit();

            for elem in &mut arr[..] {
                *elem = MaybeUninit::new(SlabBlock::new());
            }

            transmute::<_, [_; $size]>(arr)
        }
    };
}

/// Returned if out of memory.
#[derive(Debug, Clone, Copy)]
pub struct GenericSlabBoxErr;

/// A block on the slab used to allocate memory.
#[derive(Debug)]
pub struct SlabBlock<T> {
    data: UnsafeCell<MaybeUninit<T>>,
    next: AtomicUsize,
}

impl<T> SlabBlock<T> {
    /// Initializes a blank slab.
    pub fn new() -> Self {
        Self {
            data: UnsafeCell::new(unsafe { MaybeUninit::uninit() }),
            next: AtomicUsize::new(0),
        }
    }
}

/// A device for storage of a slab, such as an array on the stack, or a
/// heap-allocated vector.
pub trait SlabStorage {
    /// Data that the slab will manage through this storage.
    type Data;

    /// Length of this storage.
    fn len(&self) -> usize;

    /// Tries to index the storage. If the index is out of bounds, returns None.
    /// Must be consistent with `.len()`.
    fn try_at(&self, index: usize) -> Option<&SlabBlock<Self::Data>>;

    /// Tries to index the storage to get a mutable (exclusive) reference. If
    /// the index is out of bounds, returns None. Must be consistent with
    /// `.len()`.
    fn try_at_mut(
        &mut self,
        index: usize,
    ) -> Option<&mut SlabBlock<Self::Data>>;

    /// Indexes the storage. If the index is out of bounds, panic. Must be
    /// consistent with `.len()`.
    fn at(&self, index: usize) -> &SlabBlock<Self::Data> {
        let len = self.len();
        self.try_at(index).unwrap_or_else(|| {
            panic!("Invalid slab storage index {} (length is {})", index, len)
        })
    }

    /// Indexes the storage to get a mutable (exclusive) reference. If the index
    /// is out of bounds, panic. Must be consistent with `.len()`.
    fn at_mut(&mut self, index: usize) -> &mut SlabBlock<Self::Data> {
        let len = self.len();
        self.try_at_mut(index).unwrap_or_else(|| {
            panic!("Invalid slab storage index {} (length is {})", index, len)
        })
    }
}

impl<'slice, T> SlabStorage for &'slice mut [SlabBlock<T>] {
    type Data = T;

    fn len(&self) -> usize {
        self.len()
    }

    fn try_at(&self, index: usize) -> Option<&SlabBlock<Self::Data>> {
        self.get(index)
    }

    fn try_at_mut(
        &mut self,
        index: usize,
    ) -> Option<&mut SlabBlock<Self::Data>> {
        self.get_mut(index)
    }
}

/// A shared pointer data type.
pub trait SharedPtr: Deref + Clone {}

impl<T> SharedPtr for T where T: Deref + Clone {}

/// A slab allocator.
#[derive(Debug)]
struct Slab<S>
where
    S: SlabStorage,
{
    /// List of nodes to be destroyed. Ended by NULL_IDX.
    destroy_list: AtomicUsize,
    /// List of nodes that are free to be allocated. Ended by NULL_IDX.
    free_list: AtomicUsize,
    /// The storage device.
    storage: S,
}

/// Beginning and end of a freed destroy list.
#[derive(Debug, Clone, Copy)]
struct DestroyList {
    /// Node beginning (included)
    begin: usize,
    /// Node ending (included)
    end: usize,
}

impl<S> Slab<S>
where
    S: SlabStorage,
{
    /// Creates a new slab allocator, putting the correct indices on the blocks.
    fn new(storage: S) -> Self {
        let this = Self {
            destroy_list: AtomicUsize::new(NULL_IDX),
            free_list: AtomicUsize::new(0),
            storage,
        };

        let mut i = 0;
        while let Some(elem) = this.storage.try_at(i) {
            elem.next.store(i + 1, Release);
            i += 1;
        }

        if let Some(last) = i.checked_sub(1) {
            this.storage.at(last).next.store(NULL_IDX, Release);
        }

        this
    }

    /// Actually destroys the destroy list, but does not put the freed nodes in
    /// the free list. Returns the first and the last element of the list.
    /// Should only be called inside destroy and inside drop.
    fn free_destroy_list(&self, expected: usize) -> Result<DestroyList, usize> {
        if expected == NULL_IDX {
            return Err(expected);
        }

        self.destroy_list
            .compare_exchange(expected, NULL_IDX, Release, Acquire)?;

        let begin = data_bits(expected);
        let mut curr = begin;
        let mut next = begin;

        while next != NULL_IDX {
            unsafe {
                let cell = self.storage.at(next).data.get();
                (*cell).as_mut_ptr().drop_in_place();
            }
            curr = next;
            next = self.storage.at(curr).next.load(Relaxed);
        }

        Ok(DestroyList { begin, end: curr })
    }

    /// Destroy the free list and puts the new free nodes in the free list.
    /// Should Only be called if no one is allocating.
    fn destroy(&self, expected: usize) -> Result<usize, usize> {
        self.free_destroy_list(expected).map(|freed| {
            let mut free = self.free_list.load(Acquire);

            loop {
                self.storage.at(freed.end).next.store(free, Relaxed);
                let result = self.free_list.compare_exchange(
                    free,
                    freed.begin,
                    Release,
                    Acquire,
                );

                match result {
                    Ok(_) => break NULL_IDX,
                    Err(update) => free = update,
                }
            }
        })
    }

    /// GenericSlabBoxates a node. Should only be called if no one is freeing
    /// the destroy list.
    fn alloc(&self) -> Result<usize, GenericSlabBoxErr> {
        let mut free = self.free_list.load(Acquire);

        loop {
            if free == NULL_IDX {
                break Err(GenericSlabBoxErr);
            }

            let next = self.storage.at(free).next.load(Relaxed);
            let result =
                self.free_list.compare_exchange(free, next, Release, Acquire);
            match result {
                Ok(_) => break Ok(free),
                Err(update) => free = update,
            }
        }
    }

    /// Put a node in the to-be-destroyed-list, for later free. The node integer
    /// must contain epoch and index.
    fn free(&self, node: usize) {
        let mut destroy = self.destroy_list.load(Acquire);

        loop {
            if destroy != NULL_IDX && epoch_bit(destroy) != epoch_bit(node) {
                destroy = self
                    .destroy(destroy)
                    .unwrap_or_else(|new_destroy| new_destroy);
            }

            self.storage
                .at(data_bits(node))
                .next
                .store(data_bits(destroy), Relaxed);

            let result = self
                .destroy_list
                .compare_exchange(destroy, node, Release, Acquire);
            match result {
                Ok(_) => break,
                Err(update) => destroy = update,
            }
        }
    }
}

impl<S> Drop for Slab<S>
where
    S: SlabStorage,
{
    fn drop(&mut self) {
        let mut curr = data_bits(*self.destroy_list.get_mut());

        while curr != NULL_IDX {
            unsafe {
                let cell = self.storage.at(curr).data.get();
                (*cell).as_mut_ptr().drop_in_place();
            }
            curr = data_bits(*self.storage.at_mut(curr).next.get_mut());
        }
    }
}

/// Inner structure of allocator, accessed through a pointer.
#[derive(Debug)]
pub struct SlabAllocInner<S>
where
    S: SlabStorage,
{
    /// Slab which stores the possible allocations.
    slab: Slab<S>,
    /// Threads in critical sections (data/lower bits), generation parity
    /// (epoch/highest bit).
    threads_epoch: AtomicUsize,
}

impl<S> SlabAllocInner<S>
where
    S: SlabStorage,
{
    /// Creates a new initialized inner structure of an allocator.
    pub fn new(array: S) -> Self {
        Self { slab: Slab::new(array), threads_epoch: AtomicUsize::new(0) }
    }

    fn pause(&self) -> Pause<S> {
        let mut bits = self.threads_epoch.load(Acquire);
        loop {
            let max = data_bits(usize::max_value());
            if data_bits(bits) == max {
                panic!("Maximum thread number reached!");
            }
            let new_bits = set_data_bits(bits, data_bits(bits) + 1);
            let result = self
                .threads_epoch
                .compare_exchange(bits, new_bits, Release, Acquire);
            match result {
                Ok(_) => break,
                Err(update) => bits = update,
            }
        }

        Pause { alloc: self }
    }
}

/// An allocator. Manages concurrent memory.
#[derive(Debug)]
pub struct GenericSlabAlloc<S, P>
where
    S: SlabStorage,
    P: SharedPtr<Target = SlabAllocInner<S>>,
{
    /// Pointer to inner structure.
    inner: P,
}

impl<S, P> GenericSlabAlloc<S, P>
where
    S: SlabStorage,
    P: SharedPtr<Target = SlabAllocInner<S>>,
{
    /// Creates a new allocator from a given shared pointer to a inner allocator
    /// strcuture.
    pub fn new(inner: P) -> Self {
        Self { inner }
    }

    /// GenericSlabBoxates memory, returning error if no space is available.
    pub fn alloc(
        &self,
        val: S::Data,
    ) -> Result<GenericSlabBox<S, P>, GenericSlabBoxErr> {
        let pause = self.inner.pause();
        let index = self.inner.slab.alloc()?;
        unsafe {
            let cell = self.inner.slab.storage.at(index).data.get();
            (*cell).as_mut_ptr().write(val);
        }
        drop(pause);

        Ok(GenericSlabBox { index, alloc: self.inner.clone() })
    }
}

impl<S, P> Clone for GenericSlabAlloc<S, P>
where
    S: SlabStorage,
    P: SharedPtr<Target = SlabAllocInner<S>>,
{
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

/// A pause on the
#[derive(Debug)]
struct Pause<'col, S>
where
    S: SlabStorage,
{
    alloc: &'col SlabAllocInner<S>,
}

impl<'col, S> Drop for Pause<'col, S>
where
    S: SlabStorage,
{
    fn drop(&mut self) {
        let mut bits = self.alloc.threads_epoch.load(Acquire);

        loop {
            let new_bits = if data_bits(bits) == 1 {
                set_epoch_bit(bits, epoch_bit(bits) ^ 1)
            } else {
                set_data_bits(bits, data_bits(bits) + 1)
            };

            let result = self
                .alloc
                .threads_epoch
                .compare_exchange(bits, new_bits, Release, Acquire);
        }
    }
}

/// An allocation of memory.
#[derive(Debug)]
pub struct GenericSlabBox<S, P>
where
    S: SlabStorage,
    P: SharedPtr<Target = SlabAllocInner<S>>,
{
    index: usize,
    alloc: P,
}

/// An allocator with the default pointer as Arc.
#[cfg(feature = "std")]
pub type SlabAlloc<S> = GenericSlabAlloc<S, Arc<SlabAllocInner<S>>>;
#[cfg(feature = "std")]
/// A slab allocated box with the default pointer to allocator as Arc.
pub type SlabBox<S> = GenericSlabBox<S, Arc<SlabAllocInner<S>>>;

#[cfg(not(feature = "std"))]
/// An allocator with the default pointer as a plain reference.
pub type SlabAlloc<'inner, S> = GenericSlabAlloc<S, &'inner SlabAllocInner<S>>;
#[cfg(not(feature = "std"))]
/// A slab allocated box with the default pointer to allocator as a plain
/// reference.
pub type SlabBox<'inner, S> = GenericSlabBox<S, &'inner SlabAllocInner<S>>;
