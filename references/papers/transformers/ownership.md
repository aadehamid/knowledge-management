> IMPORTANT: To view this page as Markdown, append `.md` to the URL (e.g. /docs/manual/basics.md).
> For the complete Mojo documentation index, see [llms.txt](/llms.txt).

[Skip to main content](#__docusaurus_skipToContent_fallback)

[Mojo is now open source! Click to learn more.](https://www.modular.com/blog/mojo-open-source)

[![Mojo](/img/mojo-wordmark.svg)![Mojo](/img/mojo-wordmark-dark.svg)](/)[Install](/install/)[Docs](/docs/)[Packages](/packages/)[Releases](/releases/)[Community](/community/)

[GitHub](https://github.com/modular/modular)

[1.0.0](/docs/manual/values/ownership/)

* [Nightly](/nightly/docs/manual/values/ownership/)* [1.0.0](/docs/manual/values/ownership/)

Search

* [The Mojo Manual](/docs/manual/)* [Get started](/docs/manual/quickstart/)

    + [Quickstart](/docs/manual/quickstart/)+ [Get started with Mojo](/docs/manual/get-started/)+ [Tips for Python devs](/docs/manual/python-to-mojo/)+ [System requirements](/docs/requirements/)* [Language basics](/docs/manual/basics/)

      + [Overview](/docs/manual/basics/)+ [Functions](/docs/manual/functions/)+ [Variables](/docs/manual/variables/)+ [Types](/docs/manual/types/)+ [Operators](/docs/manual/operators/)+ [Control flow](/docs/manual/control-flow/)+ [Errors and context managers](/docs/manual/errors/)+ [Structs](/docs/manual/structs/)

                      + [Modules and packages](/docs/manual/packages/)* [Value ownership](/docs/manual/values/)

        + [Intro to value ownership](/docs/manual/values/)+ [Value semantics](/docs/manual/values/value-semantics/)+ [Ownership](/docs/manual/values/ownership/)+ [Lifetimes, origins, and references](/docs/manual/values/lifetimes/)* [Value lifecycle](/docs/manual/lifecycle/)

          + [Intro to value lifecycle](/docs/manual/lifecycle/)+ [Value creation](/docs/manual/lifecycle/life/)+ [Value destruction](/docs/manual/lifecycle/death/)+ [Deep dive - Instance initialization](/docs/manual/lifecycle/initialization/)* [Metaprogramming](/docs/manual/metaprogramming/)

            + [Intro](/docs/manual/metaprogramming/)+ [Compile-time evaluation](/docs/manual/metaprogramming/comptime-evaluation/)+ [Parameterization](/docs/manual/parameters/)+ [Traits](/docs/manual/traits/)+ [Parameterized declarations](/docs/manual/generics/)+ [Constraints](/docs/manual/metaprogramming/constraints/)+ [Materialization](/docs/manual/metaprogramming/materialization/)+ [Reflection](/docs/manual/metaprogramming/reflection/)* [Pointers](/docs/manual/pointers/)

              + [Intro to pointers](/docs/manual/pointers/)+ [Using pointers](/docs/manual/pointers/using-pointers/)* [Advanced functions](/docs/manual/functions/closures/)

                + [Closures](/docs/manual/functions/closures/)+ [Lambda expressions](/docs/manual/functions/lambda/)* [Python and C interop](/docs/manual/python/)

                  + [Introduction](/docs/manual/python/)+ [Calling Python from Mojo](/docs/manual/python/python-from-mojo/)+ [Calling Mojo from Python](/docs/manual/python/mojo-from-python/)+ [Python types](/docs/manual/python/types/)+ [Calling C from Mojo](/docs/manual/c-ffi/)* [Tools](/docs/tools/compilation/)

                    + [Compilation targets](/docs/tools/compilation/)+ [Feature toggles](/docs/tools/feature-toggles/)+ [Debugging](/docs/tools/debugging/)+ [Testing](/docs/tools/testing/)+ [Jupyter notebooks](/docs/tools/notebooks/)+ [Mojo AI skills](/docs/tools/skills/)+ [Packaging](/docs/tools/packaging/)

* * /* [Docs](/docs/)* /* [Manual](/docs/manual/)* /* Value ownership* /* Ownership
Version: 1.0.0

On this page

> For the complete Mojo documentation index, see [llms.txt](/llms.txt). Markdown versions of all pages are available by appending .md to any URL (e.g. /docs/manual/basics.md).

# Ownership

A challenge you might face when using some programming languages is that you
must manually allocate and deallocate memory. When multiple parts of the
program need access to the same memory, it becomes difficult to keep track of
who "owns" a value and determine when is the right time to deallocate it. If
you make a mistake, it may result in a "use-after-free" error, a "double free"
error, or a "leaked memory" error, any one of which can be catastrophic.

Mojo helps avoid these errors by ensuring there is only one variable that owns
each value at a time, while still allowing you to share references with other
functions. When the life span of the owner ends, Mojo [destroys the
value](/docs/manual/lifecycle/death/). Programmers are still responsible for
making sure any type that allocates resources (including memory) also
deallocates those resources in its destructor. Mojo's ownership system ensures
that destructors are called promptly.

On this page, we'll explain the rules that govern this ownership model, and how
to specify different argument conventions that define how values are passed into
functions.

## Ownership summary[​](#ownership-summary "Direct link to Ownership summary")

The fundamental rules that make Mojo's ownership model work are the following:

* Every value has only one owner at a time.
* When the lifetime of the owner ends, Mojo destroys the value.
* If there are existing references to a value, Mojo extends the lifetime of
  the owner.

### Variables and references[​](#variables-and-references "Direct link to Variables and references")

A variable *owns* its value. A struct owns its fields.

A *reference* allows you to access a value owned by another variable. A
reference has either mutable access or immutable access to that value.

Mojo references are created when you call a function: function arguments are
passed as mutable or immutable references. A function can return a
reference instead of returning a value. To capture a returned reference, you
can use a reference binding:

```
ref value_ref = list[0]
```

## Argument conventions[​](#argument-conventions "Direct link to Argument conventions")

In all programming languages, code quality and performance is heavily dependent
upon how functions treat argument values. That is, whether a value received by
a function is a unique value or a reference, and whether it's mutable or
immutable, has a series of consequences that define the readability,
performance, and safety of the language.

In Mojo, we want to provide full [value
semantics](/docs/manual/values/value-semantics/) by default, which provides
consistent and predictable behavior. But as a systems programming language, we
also need to offer full control over memory optimizations, which generally
requires reference semantics. The trick is to introduce reference semantics in
a way that ensures all code is memory safe by tracking the lifetime of every
value and destroying each one at the right time (and only once). All of this is
made possible in Mojo through the use of argument conventions that ensure every
value has only one owner at a time.

An argument convention specifies whether an argument is mutable or immutable,
and whether the function owns the value. Each convention is defined by a
keyword at the beginning of an argument declaration:

* default: The function receives an **immutable reference**. This means the
  function can read the original value (it's *not* a copy), but it can't
  mutate (modify) it.
* `mut`: The function receives a **mutable reference**. This means the
  function can read and mutate the original value (it's *not* a copy).
* `var`: The function takes **ownership** of a value. This means the function
  has exclusive ownership of the argument. The caller might choose to transfer
  ownership of an existing value to this function, but that's not always what
  happens. The callee might receive a newly-created value, or a copy of an
  existing value.
* `ref`: The function gets a reference with a parametric mutability: that is,
  it follows the mutability of the referenced value.
  `ref` arguments are an advanced topic, and they're described in more detail in
  [Lifetimes, origins, and references](/docs/manual/values/lifetimes/).
* `out`: A special convention used for the `self` argument in
  [constructors](/docs/manual/lifecycle/life/#constructor) and for
  [named results](/docs/manual/functions/#named-results). An `out`
  argument is uninitialized at the beginning of the function, and must be
  initialized before the function returns. Although `out` arguments show up in
  the argument list, they're never passed in by the caller.
* `deinit`: A special convention used in the destructor and consuming-move
  lifecycle methods. A `deinit` argument is initialized at the beginning of the
  function, and uninitialized when the function returns.

For example, this function has one argument that's a mutable
reference and one that's immutable:

```
def add(mut x: Int, y: Int):

x += y

def main():

var a = 1

var b = 2

add(a, b)

print(a)  # 3
```

You've probably already seen some function arguments that don't declare a
convention. By default, all arguments use the default convention of an
immutable read-only reference. In the following sections, we'll explain
each of these conventions in more detail.

## Immutable arguments (default)[​](#immutable-arguments-default "Direct link to Immutable arguments (default)")

The default convention is an immutable read-only reference. The callee
receives an immutable reference to the argument value.

For example:

```
def print_list(list: List[Int]):

print(list.__str__())

def main():

var values = [1, 2, 3, 4]

print_list(values)
```

```
[1, 2, 3, 4]
```

Here the `print_list()` function can read from the `list` argument, but not
mutate it. `list` is a reference to `values` in the `main()` function, not a
copy.

In general, passing an immutable reference is much more efficient
when handling large or expensive-to-copy values, because the copy constructor
and destructor aren't invoked for a default (immutable reference) argument.

### Compared to C++ and Rust[​](#compared-to-c-and-rust "Direct link to Compared to C++ and Rust")

Mojo's default argument convention is similar in some ways to passing an
argument by `const&` in C++, which also avoids a copy of the value and disables
mutability in the callee. However, the default convention differs from
`const&` in C++ in two important ways:

* The Mojo compiler implements a lifetime checker that ensures that values are
  not destroyed when there are outstanding references to those values.
* Small values like `Int`, `Float`, and `SIMD` are always passed in
  machine registers. This provides a significant performance enhancement
  compared to languages like C++ and Rust.

The major difference between Rust and Mojo is that Mojo doesn't require a
sigil on the caller side to pass by immutable reference. Also, Mojo is more
efficient when passing small values, and Rust defaults to moving values
instead of passing them around as a read-only reference. These policy and
syntax decisions allow Mojo to provide an easier-to-use programming model.

## Mutable arguments (`mut`)[​](#mutable-arguments-mut "Direct link to mutable-arguments-mut")

If you'd like your function to receive a **mutable reference**, add the `mut`
keyword in front of the argument name. You can think of `mut` like this: it
means any changes to the value *in*side the function are visible *out*side the
function.

For example, this `mutate()` function updates the original `list` value:

```
def print_list(list: List[Int]):

print(list.__str__())

def mutate(mut l: List[Int]):

l.append(5)

def main():

var values = [1, 2, 3, 4]

mutate(values)

print_list(values)
```

```
[1, 2, 3, 4, 5]
```

That behaves like an optimized replacement for this:

```
def print_list(list: List[Int]):

print(list.__str__())

def mutate_copy(l: List[Int]) -> List[Int]:

# def creates an implicit copy of the list because it's mutated

l.append(5)

return l

def main():

var values = [1, 2, 3, 4]

values = mutate_copy(values)

print_list(values)
```

```
[1, 2, 3, 4, 5]
```

Although the code using `mut` isn't that much shorter, it's more memory
efficient because it doesn't make a copy of the value.

However, remember that the values passed as `mut` must already be mutable.
For example, if you try to take an immutable reference and pass it to another
function as `mut`, you'll get a compiler error because Mojo can't form a
mutable reference from an immutable reference.

You can't define [default
values](/docs/manual/functions/#optional-arguments) for `mut`
arguments.

### Argument exclusivity[​](#argument-exclusivity "Direct link to Argument exclusivity")

Mojo enforces *argument exclusivity* for mutable references. This means that if
a function receives a mutable reference to a value (such as an `mut` argument),
it can't receive any other references to the same value—mutable or immutable.
That is, a mutable reference can't have any other references that *alias* it.

For example, consider the following code example:

```
def append_twice(mut s: String, other: String):

# Mojo knows 's' and 'other' can't be the same string.

s += other

s += other

def invalid_access():

var my_string = "o"  # Create a run-time String value

# error: passing `my_string` mut is invalid since it's also passed

# as an immutable reference

append_twice(my_string, my_string)

print(my_string)
```

This code is confusing because the user might expect the output to be `ooo`,
but since the first addition mutates both `s` and `other`, the actual output
would be `oooo`. Enforcing exclusivity of mutable references not only prevents
coding errors, it also allows the Mojo compiler to optimize code in some cases.

One way to avoid this issue when you do need both a mutable and an immutable
reference (or need to pass the same value to two arguments) is to make a copy:

```
def valid_access():

var my_string = "o"           # Create a run-time String value

var other_string = my_string  # Create a copy of the String value

append_twice(my_string, other_string)

print(my_string)
```

Note that argument exclusivity isn't enforced for register-passable trivial
types (like `Int` and `Bool`) as they're always passed by copy. When
passing the same value into two `Int` arguments, the callee receives two
copies of the value.

## Transfer arguments (`var` and `^`)[​](#transfer-arguments-var-and- "Direct link to transfer-arguments-var-and-")

If you want your function to take *ownership* of a value, add the `var`
keyword before the argument name.

This convention is often combined with using the postfix `^` transfer sigil
on an argument at the call site.

When using a variable, transferring a value leaves the original variable
uninitialized. You can't use the variable after the transfer until you
assign it a new value of the original type.

### Transferring with `var`[​](#transferring-with-var "Direct link to transferring-with-var")

`var` behaves differently depending on whether the caller uses the `^`
transfer sigil and whether the value conforms to `Copyable`.

The `var` keyword doesn't guarantee that the function receives *the original
value*. It guarantees only that the function receives *ownership of a
value*. That happens in one of three ways:

* **Value transfer**: The caller uses the `^` transfer sigil. This
  transfers the value, leaving the original variable uninitialized. The function
  argument receives ownership.
* **Copying**: Without the transfer sigil, Mojo copies the value. If the
  type isn't `Copyable`, this produces a compile-time error.
* **Newly created value**: The caller passes a newly created value,
  such as the result of a function call. In this case, no variable owns the
  value, so ownership transfers directly to the callee. For example:

  ```
  def take(var s: String):

  pass

  def main():

  take("A brand-new String!")
  ```

The following code works by making a copy of the string, because `take_text()`
uses the `var` convention, and the caller doesn't include the transfer sigil:

```
def take_text(var text: String):

text += "!"

print(text)

def main():

var message = "Hello"  # Create a run-time String value

take_text(message)

print(message)
```

```
Hello!

Hello
```

However, if you add the `^` transfer sigil when calling `take_text()`, the
compiler complains about `print(message)`, because at that point, the `message`
variable is no longer initialized. That is, this version doesn't compile:

```
def main():

var message = "Hello"  # Create a run-time String value

take_text(message^)

print(message)  # error: use of uninitialized value 'message'
```

This is a critical feature of Mojo's lifetime checker, because it ensures that
no two variables have ownership of the same value. To fix the error, you must
not use the `message` variable after you end its lifetime with the `^` transfer
sigil. So here is the corrected code:

```
def take_text(var text: String):

text += "!"

print(text)

def main():

var message = "Hello"  # Create a run-time String value

take_text(message^)
```

```
Hello!
```

Regardless of how it receives the value, when the function declares an argument
as `var`, it's certain that it has unique mutable access to that value.
Because the value is owned, the value is destroyed when the function
exits—unless the function transfers the value elsewhere.

For example, in the following example, `add_to_list()` takes a string and
appends it to the list. Ownership of the string is transferred to the list, so
it's not destroyed when the function exits. On the other hand,
`consume_string()` doesn't transfer its `var` value out, so the value is
destroyed at the end of the function.

```
def add_to_list(var name: String, mut list: List[String]):

list.append(name^)

# name is uninitialized, nothing to destroy

def consume_string(var s: String):

print(s)

# s is destroyed here
```

### Transfer implementation details[​](#transfer-implementation-details "Direct link to Transfer implementation details")

In Mojo, you shouldn't conflate "ownership transfer" with a "move
operation"—these aren't strictly the same thing.

There are multiple ways that Mojo transfers ownership of a value:

* If a type implements the [move
  constructor](/docs/manual/lifecycle/life/#move-constructor),
  `__init__(take=)`, Mojo may invoke this method *if* a value of that type is
  transferred into a function as a `var` argument, *and* the original
  variable's lifetime ends at the same point (with or without use of the `^`
  transfer sigil).
* In some cases, Mojo optimizes away the move operation entirely, leaving the
  value in the same memory location but updating its ownership. In these cases,
  a value transfers without invoking either the copy or move
  constructors.

In order for the `var` convention to work *without* the transfer sigil, the
value type must be copyable (via `__init__(out self, *, copy: Self)`).

[Edit this page](https://github.com/modular/modular/edit/main/mojo/docs/manual/values/ownership.mdx)

[Previous

Value semantics](/docs/manual/values/value-semantics/)[Next

Lifetimes, origins, and references](/docs/manual/values/lifetimes/)

[Edit this page](https://github.com/modular/modular/edit/main/mojo/docs/manual/values/ownership.mdx)

* [Ownership summary](#ownership-summary)
  + [Variables and references](#variables-and-references)* [Argument conventions](#argument-conventions)* [Immutable arguments (default)](#immutable-arguments-default)
      + [Compared to C++ and Rust](#compared-to-c-and-rust)* [Mutable arguments (`mut`)](#mutable-arguments-mut)
        + [Argument exclusivity](#argument-exclusivity)* [Transfer arguments (`var` and `^`)](#transfer-arguments-var-and-)
          + [Transferring with `var`](#transferring-with-var)+ [Transfer implementation details](#transfer-implementation-details)

[![Mojo](/img/mojo-wordmark.svg)](/)[Releases](/releases/)[Roadmap](/docs/roadmap/)[License](https://github.com/modular/modular/blob/main/LICENSE)[Contributing](https://github.com/modular/modular/blob/main/CONTRIBUTING.md)[Install now](/install/)