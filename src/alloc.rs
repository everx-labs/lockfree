use core::{
    cell::UnsafeCell,
    mem::{forget, size_of, MaybeUninit},
    ops::{Deref, DerefMut},
    sync::atomic::{
        AtomicUsize,
        Ordering::{self, *},
    },
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
#[macro_export]
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

/// Called when maximum number of threads is reached.
#[cold]
#[inline(never)]
fn too_much_threads() -> ! {
    panic!("Maximum thread number reached!");
}

/// Returned if out of memory.
#[derive(Debug, Clone, Copy)]
pub struct AllocErr;

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
            data: UnsafeCell::new(MaybeUninit::uninit()),
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
        (**self).len()
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

/// A slab allocator.
#[derive(Debug)]
pub struct SlabInner<S>
where
    S: SlabStorage,
{
    /// List of nodes to be destroyed. Ended by NULL_IDX.
    destroy_list: AtomicUsize,
    /// List of nodes that are free to be allocated. Ended by NULL_IDX.
    free_list: AtomicUsize,
    /// The storage device.
    storage: S,
    /// Threads in critical sections (data/lower bits), generation parity
    /// (epoch/highest bit).
    threads_epoch: AtomicUsize,
}

/// Beginning and end of a freed destroy list.
#[derive(Debug, Clone, Copy)]
struct DestroyList {
    /// Node beginning (included)
    begin: usize,
    /// Node ending (included)
    end: usize,
}

impl<S> SlabInner<S>
where
    S: SlabStorage,
{
    /// Creates a new slab allocator, putting the correct indices on the blocks.
    pub fn new(storage: S) -> Self {
        let this = Self {
            destroy_list: AtomicUsize::new(NULL_IDX),
            free_list: AtomicUsize::new(0),
            storage,
            threads_epoch: AtomicUsize::new(0),
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

    /// SlabBoxates a node. Should only be called if no one is freeing
    /// the destroy list.
    fn alloc(&self) -> Result<usize, AllocErr> {
        let mut free = self.free_list.load(Acquire);

        loop {
            if free == NULL_IDX {
                break Err(AllocErr);
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

    /// Pauses the slab deallocation.
    fn pause(&self) -> Pause<S> {
        let mut bits = self.threads_epoch.load(Acquire);
        loop {
            let max = data_bits(usize::max_value());
            if data_bits(bits) == max {
                too_much_threads()
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

        Pause { slab: self }
    }
}

impl<S> Drop for SlabInner<S>
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

/// A shared pointer data type.
pub trait SharedSlabPtr
where
    Self: Deref<Target = SlabInner<<Self as SharedSlabPtr>::Storage>>,
    Self: Clone,
{
    type Storage: SlabStorage;
}

/// An allocator. Manages concurrent memory.
#[derive(Debug)]
pub struct SlabAlloc<P>
where
    P: SharedSlabPtr,
{
    /// Pointer to inner slab.
    inner: P,
}

impl<P> SlabAlloc<P>
where
    P: SharedSlabPtr,
{
    /// Creates a new allocator from a given shared pointer to a inner allocator
    /// strcuture.
    pub fn new(inner: P) -> Self {
        Self { inner }
    }

    /// SlabBoxates memory, returning error if no space is available.
    pub fn alloc(
        &self,
        val: <P::Storage as SlabStorage>::Data,
    ) -> Result<SlabBox<P>, AllocErr> {
        let pause = self.inner.pause();
        let index = self.inner.alloc()?;
        unsafe {
            let cell = self.inner.storage.at(index).data.get();
            (*cell).as_mut_ptr().write(val);
        }
        drop(pause);

        Ok(SlabBox { index, alloc: self.clone() })
    }
}

impl<P> Clone for SlabAlloc<P>
where
    P: SharedSlabPtr,
{
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

/// A pause on the
#[derive(Debug)]
struct Pause<'slab, S>
where
    S: SlabStorage,
{
    slab: &'slab SlabInner<S>,
}

impl<'slab, S> Drop for Pause<'slab, S>
where
    S: SlabStorage,
{
    fn drop(&mut self) {
        let mut bits = self.slab.threads_epoch.load(Acquire);

        loop {
            let max = data_bits(usize::max_value());

            if data_bits(bits) != max {
                too_much_threads()
            }

            let new_bits = if data_bits(bits) == 1 {
                set_epoch_bit(bits, epoch_bit(bits) ^ 1)
            } else {
                set_data_bits(bits, data_bits(bits) + 1)
            };

            let result = self
                .slab
                .threads_epoch
                .compare_exchange(bits, new_bits, Release, Acquire);

            match result {
                Ok(_) if data_bits(bits) == 1 => {
                    let _ = self.slab.destroy(epoch_bit(bits));
                    self.slab.threads_epoch.fetch_sub(1, Release);
                    break;
                },

                Ok(_) => break,

                Err(update) => bits = update,
            }
        }
    }
}

/// An allocation of memory.
#[derive(Debug)]
pub struct SlabBox<P>
where
    P: SharedSlabPtr,
{
    /// Index in the slab of this allocation.
    index: usize,
    /// Original allocator.
    alloc: SlabAlloc<P>,
}

impl<P> SlabBox<P>
where
    P: SharedSlabPtr,
{
    /// Make this box atomic and thus inner-mutable (i.e. the pointer can be
    /// changed but it is not seen as mutable by the type system.
    pub fn atomic(self) -> AtomicBox<P> {
        let atomic = AtomicBox {
            index: AtomicUsize::new(self.index),
            alloc: self.alloc.clone(),
        };

        forget(self);

        atomic
    }
}

impl<P> Drop for SlabBox<P>
where
    P: SharedSlabPtr,
{
    fn drop(&mut self) {
        let pause = self.alloc.inner.pause();
        self.alloc.inner.free(self.index);
        drop(pause);
    }
}

impl<P> Deref for SlabBox<P>
where
    P: SharedSlabPtr,
{
    type Target = <P::Storage as SlabStorage>::Data;

    fn deref(&self) -> &Self::Target {
        unsafe {
            let cell = self.alloc.inner.storage.at(self.index).data.get();
            &(*(*cell).as_ptr())
        }
    }
}

impl<P> DerefMut for SlabBox<P>
where
    P: SharedSlabPtr,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            let cell = self.alloc.inner.storage.at(self.index).data.get();
            &mut (*(*cell).as_mut_ptr())
        }
    }
}

/// An atomic slab allocated box.
pub struct AtomicBox<P>
where
    P: SharedSlabPtr,
{
    /// Atomically stored index of the slab node.
    index: AtomicUsize,
    /// Original allocator.
    alloc: SlabAlloc<P>,
}

impl<P> AtomicBox<P>
where
    P: SharedSlabPtr,
{
    pub fn load(&self, ordering: Ordering) -> LoadedBox<P> {
        unimplemented!()
    }
}

impl<P> Drop for AtomicBox<P>
where
    P: SharedSlabPtr,
{
    fn drop(&mut self) {
        let pause = self.alloc.inner.pause();
        self.alloc.inner.free(*self.index.get_mut());
        drop(pause);
    }
}

/// A slab allocated box loaded from `AtomicBox`.
pub struct LoadedBox<P>
where
    P: SharedSlabPtr,
{
    /// Index of the slab node.
    index: usize,
    /// Original allocator.
    alloc: SlabAlloc<P>,
}

impl<P> Deref for LoadedBox<P>
where
    P: SharedSlabPtr,
{
    type Target = <P::Storage as SlabStorage>::Data;

    fn deref(&self) -> &Self::Target {
        unsafe {
            let cell = self.alloc.inner.storage.at(self.index).data.get();
            &(*(*cell).as_ptr())
        }
    }
}

/// A slab allocator with owned pointer and slab storage.
#[cfg(feature = "std")]
pub type OwnedSlabAlloc<T> = SlabAlloc<Arc<SlabInner<Box<[SlabBlock<T>]>>>>;

/// A slab allocated box with owned pointer and slab storage.
#[cfg(feature = "std")]
pub type OwnedSlabBox<T> = SlabBox<Arc<SlabInner<Box<[SlabBlock<T>]>>>>;

/// An atomic slab allocated box with owned pointer and slab storage.
#[cfg(feature = "std")]
pub type OwnedAtomicBox<T> = AtomicBox<Arc<SlabInner<Box<[SlabBlock<T>]>>>>;

/// A slab allocator with borrowed pointer and slab storage.
pub type RefSlabAlloc<'slab, 'inner, T> =
    SlabAlloc<&'inner SlabInner<&'slab [SlabBlock<T>]>>;

/// A slab allocated box with borrowed pointer and slab storage.
pub type RefSlabBox<'slab, 'inner, T> =
    SlabBox<&'inner SlabInner<&'slab [SlabBlock<T>]>>;

/// An atomic slab allocated box with borrowed pointer and slab storage.
pub type RefAtomicBox<'slab, 'inner, T> =
    AtomicBox<&'inner SlabInner<&'slab [SlabBlock<T>]>>;
