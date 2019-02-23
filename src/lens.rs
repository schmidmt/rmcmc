/// A container for getting and setting a value in a struct
///
/// # Parameters
/// `T`: The type of the value
/// `S`: The type of the struct on which to operate
///
/// # Example
/// ```
/// #[macro_use] extern crate rmcmc;
/// # use rmcmc::lens::*;
///
/// # fn main() {
/// struct Foo {
///     pub bar: i32,
/// }
///
/// let len = make_lens!(Foo, i32, bar);
/// let a = Foo { bar: 1 };
///
/// assert!(len.get(&a) == 1);
///
/// let b = len.set(&a, 2);
/// assert!(b.bar == 2);
/// # }
/// ```

pub struct Lens<T, S> {
    // Getter function
    pub get_func: fn(&S) -> T,
    // Setter function
    pub set_func: fn(&S, T) -> S,
}

impl<T, S> Clone for Lens<T, S> {
    fn clone(&self) -> Self {
        Lens { ..*self }
    }
}

impl<T, S> Lens<T, S> {
    pub fn new(get: fn(&S) -> T, set: fn(&S, T) -> S) -> Self {
        Lens {
            get_func: get,
            set_func: set,
        }
    }

    pub fn set(&self, s: &S, x: T) -> S {
        (self.set_func)(&s, x)
    }

    pub fn get(&self, s: &S) -> T {
        (self.get_func)(&s)
    }

    pub fn set_in_place(&self, s: &mut S, x: T) {
        *s = self.set(s, x);
    }
}

#[macro_export]
macro_rules! make_lens {
    ($kind: ident, $ptype: ty, $param: ident) => {
        Lens::new(
            |s: &$kind| (*s).$param,
            |s: &$kind, x: $ptype| $kind { $param: x, ..*s },
        )
    };
}

#[macro_export]
macro_rules! make_lens_clone {
    ($kind: ident, $ptype: ty, $param: ident) => {
        Lens::new(
            |s: &$kind| (*s).$param.clone(),
            |s: &$kind, x: $ptype| $kind { $param: x, ..*s },
        )
    };
}

#[cfg(test)]
mod tests {
    //extern crate assert;
    extern crate test;
    // use self::test::Bencher;

    use super::*;

    #[test]
    fn simple() {
        struct Foo {
            pub bar: i32,
        }

        let len: Lens<i32, Foo> = Lens::new(
            |x: &Foo| (*x).bar,
            |x: &Foo, y: i32| Foo { bar: y, ..*x },
        );

        let a = Foo { bar: 1 };
        assert!(len.get(&a) == 1);

        let b = len.set(&a, 2);
        assert!(b.bar == 2);
    }

    #[test]
    fn simple_macro() {
        struct Foo {
            pub bar: i32,
        }

        let len = make_lens!(Foo, i32, bar);
        let a = Foo { bar: 1 };

        assert!(len.get(&a) == 1);

        let b = len.set(&a, 2);
        assert!(b.bar == 2);
    }
}
