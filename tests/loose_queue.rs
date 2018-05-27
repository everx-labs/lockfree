extern crate lockfree;

use lockfree::prelude::*;
use std::{sync::Arc, thread};

#[test]
fn debug_iter() {
    let queue = LooseQueue::new();
    queue.push(9);
    queue.push(8);
    queue.push(7);
    assert_eq!(format!("{:?}", queue), "front <= 9 <= 8 <= 7 <= back");
}

#[test]
fn into_and_from_iter() {
    let answer = vec![9, 8, 7];
    let queue = answer.clone().into_iter().collect::<LooseQueue<_>>();
    let result = queue.into_iter().collect::<Vec<_>>();
    assert_eq!(result, answer);
}

#[test]
fn single_threaded_order() {
    let queue = LooseQueue::new();
    assert_eq!(queue.pop(), None);
    assert_eq!(queue.pop(), None);
    queue.push(3);
    queue.push(5);
    assert_eq!(queue.pop(), Some(3));
    queue.push(7);
    queue.push(8);
    assert_eq!(queue.pop(), Some(5));
    assert_eq!(queue.pop(), Some(7));
    assert_eq!(queue.pop(), Some(8));
    assert_eq!(queue.pop(), None);
    assert_eq!(queue.pop(), None);
    queue.push(2);
    queue.push(0);
    assert_eq!(queue.pop(), Some(2));
    assert_eq!(queue.pop(), Some(0));
    assert_eq!(queue.pop(), None);
    assert_eq!(queue.pop(), None);
}

#[test]
fn multithreaded() {
    generic_multithreaded(20, 800, 55);
    generic_multithreaded(100, 10000, 87);
}

fn generic_multithreaded(nthread: usize, niter: usize, nmod: usize) {
    let queue = Arc::new(LooseQueue::new());
    let mut handles = Vec::with_capacity(nthread);
    for i in 0..nthread {
        let queue = queue.clone();
        handles.push(thread::spawn(move || {
            for j in 0..niter {
                let val = (i * nthread) + j;
                queue.push(val);
                if val % nmod == 0 {
                    queue.pop();
                }
            }
        }));
    }
    for handle in handles {
        handle.join().unwrap();
    }
    let expected = niter * nthread - niter * nthread / nmod;
    let mut res = 0;
    while let Some(_) = queue.pop() {
        res += 1;
    }
    assert!(res >= expected);
}