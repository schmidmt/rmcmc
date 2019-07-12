/// A container for getting and setting a value in a struct
///
/// # Parameters
/// `T`: The type of the value
/// `S`: The type of the struct on which to operate
///
/// # Example
/// ```
/// #[macro_use] extern crate rmcmc;
/// # use rmcmc::*;
///
/// # fn main() {
/// struct Foo {
///     pub bar: i32,
/// }
///
/// let len = make_lens!(Foo, i32, bar);
/// let a = Foo { bar: 1 };
///
/// assert_eq!(len.get(&a), &1_i32);
///
/// let b = len.set(&a, 2);
/// assert_eq!(b.bar, 2);
/// # }
/// ```
pub struct Lens<T, S> {
    // Getter function
    get_func: fn(&S) -> &T,
    // Setter function
    set_func: fn(&S, T) -> S,
}

impl<T, S> Clone for Lens<T, S> {
    fn clone(&self) -> Self {
        Lens { ..*self }
    }
}

impl<T, S> Lens<T, S> {
    /// Create a new lens with getter and setter functions.
    pub fn new(get: fn(&S) -> &T, set: fn(&S, T) -> S) -> Self {
        Lens {
            get_func: get,
            set_func: set,
        }
    }

    /// Set a value `x` in the struct `s`
    pub fn set(&self, s: &S, x: T) -> S {
        (self.set_func)(&s, x)
    }

    /// Retrieve the value from `s`
    pub fn get<'a>(&self, s: &'a S) -> &'a T {
        (self.get_func)(&s)
    }

    /// Set the value `x` into `s` in-place (no copy).
    pub fn set_in_place(&self, s: &mut S, x: T) {
        *s = self.set(s, x);
    }
}

/// Make a lens from a simple structure (inner type must implement Copy)
/// # Arguments
/// * `kind` - Outer Struct Type
/// * `ptype` - Inner Type
/// * `param` - Inner value's name
#[macro_export]
macro_rules! make_lens {
    ($kind: ident, $ptype: ty, $param: ident) => {
        Lens::new(
            |s: &$kind| &(s.$param),
            |s: &$kind, x: $ptype| $kind { $param: x, ..*s },
        )
    };
}

/// Make a lens from a simple structure (inner type must implement Clone)
/// # Arguments
/// * `kind` - Outer Struct Type
/// * `ptype` - Inner Type
/// * `param` - Inner value's name
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
    use super::*;

    #[test]
    fn simple() {
        struct Foo {
            pub bar: i32,
        }

        let len: Lens<i32, Foo> = Lens::new(
            |x: &Foo| &(*x).bar,
            |x: &Foo, y: i32| Foo { bar: y, ..*x },
        );

        let a = Foo { bar: 1 };
        assert_eq!(len.get(&a), &1i32);

        let b = len.set(&a, 2);
        assert_eq!(b.bar, 2i32);
    }

    #[test]
    fn simple_macro() {
        struct Foo {
            pub bar: i32,
        }

        let len = make_lens!(Foo, i32, bar);
        let a = Foo { bar: 1 };

        assert_eq!(len.get(&a).clone(), 1);

        let b = len.set(&a, 2_i32);
        assert_eq!(b.bar, 2);
    }
}
