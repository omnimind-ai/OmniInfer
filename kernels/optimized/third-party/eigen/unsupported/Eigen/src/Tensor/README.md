# Eigen Tensors {#eigen_tensors}

Tensors are multidimensional arrays of elements. Elements are typically scalars,
but more complex types such as strings are also supported.

The Tensor module is part of Eigen's unsupported modules. While it is actively
used in production (e.g. in TensorFlow), its API may change without notice.

To use the Tensor module, include the following header:

```cpp
#include <unsupported/Eigen/Tensor>
```

## Quick Start

```cpp
#include <unsupported/Eigen/Tensor>
#include <iostream>

int main() {
  // Create a 3x4 matrix as a rank-2 tensor.
  Eigen::Tensor<float, 2> a(3, 4);
  a.setRandom();

  // Create another tensor and compute their element-wise sum.
  Eigen::Tensor<float, 2> b(3, 4);
  b.setConstant(1.0f);
  Eigen::Tensor<float, 2> c = a + b;

  // Reduce: compute the sum of all elements.
  Eigen::Tensor<float, 0> total = c.sum();
  std::cout << "Sum of all elements: " << total() << "\n";

  // Reshape and broadcast.
  Eigen::Tensor<float, 2> d = c.reshape(Eigen::array<Eigen::Index, 2>{{1, 12}})
                                .broadcast(Eigen::array<Eigen::Index, 2>{{3, 1}});
  std::cout << "d has shape: " << d.dimension(0) << " x " << d.dimension(1) << "\n";
  return 0;
}
```

## Tensor Classes

You can manipulate a tensor with one of the following classes.  They all are in
the namespace `::Eigen`.

### Class Tensor\<Scalar, NumIndices, Options, IndexType\>

This is the class to use to create a tensor and allocate memory for it.

Template parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `Scalar` | Element type (e.g. `float`, `int`, `std::string`) | (required) |
| `NumIndices` | Rank (number of dimensions) | (required) |
| `Options` | `ColMajor` (0) or `RowMajor` | `0` (`ColMajor`) |
| `IndexType` | Type used for indexing (e.g. `int`, `long`) | `Eigen::DenseIndex` |

Tensors of this class are resizable.  For example, if you assign a tensor of a
different size to a Tensor, that tensor is resized to match its new value.

#### Constructor Tensor\<Scalar, NumIndices\>(size0, size1, ...)

Constructor for a Tensor.  The constructor must be passed `NumIndices` integers
indicating the sizes of the instance along each of the dimensions.

```cpp
// Create a tensor of rank 3 of sizes 2, 3, 4.  This tensor owns
// memory to hold 24 floating point values (24 = 2 x 3 x 4).
Tensor<float, 3> t_3d(2, 3, 4);

// Resize t_3d by assigning a tensor of different sizes, but same rank.
t_3d = Tensor<float, 3>(3, 4, 3);
```

#### Constructor Tensor\<Scalar, NumIndices\>(size_array)

Constructor where the sizes for the constructor are specified as an array of
values instead of an explicit list of parameters.  The array type to use is
`Eigen::array<Eigen::Index, NumIndices>`.  The array can be constructed
automatically from an initializer list.

```cpp
// Create a tensor of strings of rank 2 with sizes 5, 7.
Tensor<string, 2> t_2d({5, 7});
```

### Class TensorFixedSize\<Scalar, Sizes\<size0, size1, ...\>, Options, IndexType\>

Class to use for tensors of fixed size, where the size is known at compile
time.  Fixed sized tensors can provide very fast computations because all their
dimensions are known by the compiler.  FixedSize tensors are not resizable.

If the total number of elements in a fixed size tensor is small enough the
tensor data is held onto the stack and does not cause heap allocation and free.

```cpp
// Create a 4 x 3 tensor of floats.
TensorFixedSize<float, Sizes<4, 3>> t_4x3;
```

### Class TensorMap\<Tensor\<Scalar, NumIndices, Options\>\>

This is the class to use to create a tensor on top of memory allocated and
owned by another part of your code.  It allows to view any piece of allocated
memory as a `Tensor`.  Instances of this class do not own the memory where the
data are stored.

A `TensorMap` is not resizable because it does not own the memory where its data
are stored.

An optional alignment template parameter controls whether Eigen can assume
the data pointer is aligned: `TensorMap<Tensor<float, 2>, Aligned>`.

#### Constructor TensorMap\<Tensor\<Scalar, NumIndices\>\>(data, size0, size1, ...)

Constructor for a TensorMap.  The constructor must be passed a pointer to the
storage for the data, and `NumIndices` size attributes.  The storage has to be
large enough to hold all the data.

```cpp
// Map a tensor of ints on top of stack-allocated storage.
int storage[128];  // 2 x 4 x 2 x 8 = 128
TensorMap<Tensor<int, 4>> t_4d(storage, 2, 4, 2, 8);

// The same storage can be viewed as a different tensor.
// You can also pass the sizes as an array.
TensorMap<Tensor<int, 2>> t_2d(storage, 16, 8);

// You can also map fixed-size tensors.  Here we get a 1d view of
// the 2d fixed-size tensor.
TensorFixedSize<float, Sizes<4, 3>> t_4x3;
TensorMap<Tensor<float, 1>> t_12(t_4x3.data(), 12);
```

#### Class TensorRef

See **Assigning to a `TensorRef`**.

## Accessing Tensor Elements

#### Scalar tensor(index0, index1...)

Return the element at position `(index0, index1...)` in tensor
`tensor`.  You must pass as many parameters as the rank of `tensor`.
The expression can be used as an l-value to set the value of the element at the
specified position.  The value returned is of the datatype of the tensor.

```cpp
// Set the value of the element at position (0, 1, 0);
Tensor<float, 3> t_3d(2, 3, 4);
t_3d(0, 1, 0) = 12.0f;

// Initialize all elements to random values.
for (int i = 0; i < 2; ++i) {
  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 4; ++k) {
      t_3d(i, j, k) = ...some random value...;
    }
  }
}

// Print elements of a tensor.
for (int i = 0; i < 2; ++i) {
  std::cout << t_3d(i, 0, 0);
}
```

## TensorLayout

The tensor library supports 2 layouts: `ColMajor` (the default) and
`RowMajor`.

The layout of a tensor is optionally specified as the third template parameter
(`Options`). If not specified explicitly, column major is assumed.

```cpp
Tensor<float, 3, ColMajor> col_major;  // equivalent to Tensor<float, 3>
TensorMap<Tensor<float, 3, RowMajor>> row_major(data, ...);
```

All the arguments to an expression must use the same layout. Attempting to mix
different layouts will result in a compilation error.

It is possible to change the layout of a tensor or an expression using the
`swap_layout()` method.  Note that this will also reverse the order of the
dimensions.

```cpp
Tensor<float, 2, ColMajor> col_major(2, 4);
Tensor<float, 2, RowMajor> row_major(2, 4);

Tensor<float, 2> col_major_result = col_major;  // ok, layouts match
Tensor<float, 2> col_major_result = row_major;  // will not compile

// Simple layout swap
col_major_result = row_major.swap_layout();
eigen_assert(col_major_result.dimension(0) == 4);
eigen_assert(col_major_result.dimension(1) == 2);

// Swap the layout and preserve the order of the dimensions
array<int, 2> shuffle{{1, 0}};
col_major_result = row_major.swap_layout().shuffle(shuffle);
eigen_assert(col_major_result.dimension(0) == 2);
eigen_assert(col_major_result.dimension(1) == 4);
```

## Tensor Operations

The Eigen Tensor library provides a vast library of operations on Tensors:
numerical operations such as addition and multiplication, geometry operations
such as slicing and shuffling, etc.  These operations are available as methods
of the `Tensor` classes, and in some cases as operator overloads.  For example
the following code computes the elementwise addition of two tensors:

```cpp
Tensor<float, 3> t1(2, 3, 4);
t1.setRandom();
Tensor<float, 3> t2(2, 3, 4);
t2.setRandom();
// Set t3 to the element wise sum of t1 and t2
Tensor<float, 3> t3 = t1 + t2;
```

While the code above looks easy enough, it is important to understand that the
expression `t1 + t2` is not actually adding the values of the tensors.  The
expression instead constructs a "tensor operator" object of the class
`TensorCwiseBinaryOp<scalar_sum>`, which has references to the tensors
`t1` and `t2`.  This is a small C++ object that knows how to add
`t1` and `t2`.  It is only when the value of the expression is assigned
to the tensor `t3` that the addition is actually performed.  Technically,
this happens through the overloading of `operator=` in the Tensor class.

This mechanism for computing tensor expressions allows for lazy evaluation and
optimizations which are what make the tensor library very fast.

Of course, the tensor operators do nest, and the expression `t1 + t2 * 0.3f`
is actually represented with the (approximate) tree of operators:

```cpp
TensorCwiseBinaryOp<scalar_sum>(t1, TensorCwiseUnaryOp<scalar_mul>(t2, 0.3f))
```

### Tensor Operations and C++ "auto"

Because `Tensor` operations create tensor operators, the C++ `auto` keyword
does not have its intuitive meaning.  Consider these 2 lines of code:

```cpp
Tensor<float, 3> t3 = t1 + t2;
auto t4 = t1 + t2;
```

In the first line we allocate the tensor `t3` and it will contain the
result of the addition of `t1` and `t2`.  In the second line, `t4`
is actually the tree of tensor operators that will compute the addition of
`t1` and `t2`.  In fact, `t4` is *not* a tensor and you cannot get
the values of its elements:

```cpp
Tensor<float, 3> t3 = t1 + t2;
std::cout << t3(0, 0, 0);  // OK prints the value of t1(0, 0, 0) + t2(0, 0, 0)

auto t4 = t1 + t2;
std::cout << t4(0, 0, 0);  // Compilation error!
```

When you use `auto` you do not get a `Tensor` as a result but instead a
non-evaluated expression.
So only use `auto` to delay evaluation.

Unfortunately, there is no single underlying concrete type for holding
non-evaluated expressions, hence you have to use `auto` in the case when you do
want to hold non-evaluated expressions.

When you need the results of set of tensor computations you have to assign the
result to a `Tensor` that will be capable of holding onto them.  This can be
either a normal `Tensor`, a `TensorFixedSize`, or a `TensorMap` on an existing
piece of memory.  All the following will work:

```cpp
auto t4 = t1 + t2;

Tensor<float, 3> result = t4;  // Could also be: result(t4);
std::cout << result(0, 0, 0);

TensorMap<Tensor<float, 3>> result2(some_float_ptr, dim0, dim1, dim2);
result2 = t4;
std::cout << result2(0, 0, 0);

TensorFixedSize<float, Sizes<4, 4, 2>> result3 = t4;
std::cout << result3(0, 0, 0);
```

Until you need the results, you can keep the operation around, and even reuse
it for additional operations.  As long as you keep the expression as an
operation, no computation is performed.

```cpp
// One way to compute exp((t1 + t2) * 0.2f);
auto t3 = t1 + t2;
auto t4 = t3 * 0.2f;
auto t5 = t4.exp();
Tensor<float, 3> result = t5;

// Another way, exactly as efficient as the previous one:
Tensor<float, 3> result = ((t1 + t2) * 0.2f).exp();
```

### Controlling When Expressions are Evaluated

There are several ways to control when expressions are evaluated:

*   Assignment to a `Tensor`, `TensorFixedSize`, or `TensorMap`.
*   Use of the `eval()` method.
*   Assignment to a `TensorRef`.

#### Assigning to a Tensor, TensorFixedSize, or TensorMap.

The most common way to evaluate an expression is to assign it to a `Tensor`.
In the example below, the `auto` declarations make the intermediate values
"Operations", not Tensors, and do not cause the expressions to be evaluated.
The assignment to the Tensor `result` causes the evaluation of all the
operations.

```cpp
auto t3 = t1 + t2;             // t3 is an Operation.
auto t4 = t3 * 0.2f;           // t4 is an Operation.
auto t5 = t4.exp();            // t5 is an Operation.
Tensor<float, 3> result = t5;  // The operations are evaluated.
```

If you know the ranks and sizes of the Operation value you can assign the
Operation to a `TensorFixedSize` instead of a `Tensor`, which is a bit more efficient.

```cpp
// We know that the result is a 4x4x2 tensor!
TensorFixedSize<float, Sizes<4, 4, 2>> result = t5;
```

Similarly, assigning an expression to a `TensorMap` causes its evaluation.
Like tensors of type `TensorFixedSize`, a `TensorMap` cannot be resized so they have to
have the rank and sizes of the expression that are assigned to them.

#### Calling eval().

When you compute large composite expressions, you sometimes want to tell Eigen
that an intermediate value in the expression tree is worth evaluating ahead of
time.
This is done by inserting a call to the `eval()` method of the
expression Operation.

```cpp
// The previous example could have been written:
Tensor<float, 3> result = ((t1 + t2) * 0.2f).exp();

// If you want to compute (t1 + t2) once ahead of time you can write:
Tensor<float, 3> result = ((t1 + t2).eval() * 0.2f).exp();
```

Semantically, calling `eval()` is equivalent to materializing the value of
the expression in a temporary `Tensor` of the right size.
The code above in effect does:

```cpp
// .eval() knows the size!
TensorFixedSize<float, Sizes<4, 4, 2>> tmp = t1 + t2;
Tensor<float, 3> result = (tmp * 0.2f).exp();
```

Note that the return value of `eval()` is itself an Operation, so the
following code does not do what you may think:

```cpp
// Here t3 is an evaluation Operation.  t3 has not been evaluated yet.
auto t3 = (t1 + t2).eval();

// You can use t3 in another expression.  Still no evaluation.
auto t4 = (t3 * 0.2f).exp();

// The value is evaluated when you assign the Operation to a Tensor, using
// an intermediate tensor to represent t3.
Tensor<float, 3> result = t4;
```

While in the examples above calling `eval()` does not make a difference in
performance, in other cases it can make a huge difference.  In the expression
below the `broadcast()` expression causes the `X.maximum()` expression
to be evaluated many times:

```cpp
Tensor<...> X ...;
Tensor<...> Y = ((X - X.maximum(depth_dim).reshape(dims2d).broadcast(bcast))
                 * beta).exp();
```

Inserting a call to `eval()` between the `maximum()` and
`reshape()` calls guarantees that `maximum()` is only computed once and
greatly speeds-up execution:

```cpp
Tensor<...> Y =
  ((X - X.maximum(depth_dim).eval().reshape(dims2d).broadcast(bcast))
    * beta).exp();
```

In the other example below, the tensor `Y` is both used in the expression and its assignment.
This is an aliasing problem and if the evaluation is not done in the right order
Y will be updated incrementally during the evaluation
resulting in bogus results:

```cpp
 Tensor<...> Y ...;
 Y = Y / (Y.sum(depth_dim).reshape(dims2d).broadcast(bcast));
```

Inserting a call to `eval()` between the `sum()` and `reshape()`
expressions ensures that the sum is computed before any updates to `Y` are
done.

```cpp
 Y = Y / (Y.sum(depth_dim).eval().reshape(dims2d).broadcast(bcast));
```

Note that an eval around the full right hand side expression is not needed
because the generated has to compute the `i`-th value of the right hand side
before assigning it to the left hand side.

However, if you were assigning the expression value to a shuffle of `Y`
then you would need to force an eval for correctness by adding an `eval()`
call for the right hand side:

```cpp
 Y.shuffle(...) =
    (Y / (Y.sum(depth_dim).eval().reshape(dims2d).broadcast(bcast))).eval();
```

#### Assigning to a TensorRef.

If you need to access only a few elements from the value of an expression you
can avoid materializing the value in a full tensor by using a `TensorRef`.

A `TensorRef` is a small wrapper class for any Eigen Operation.  It provides
overloads for the `()` operator that let you access individual values in
the expression.
`TensorRef` is convenient, because the Operation themselves do
not provide a way to access individual elements.

```cpp
// Create a TensorRef for the expression.  The expression is not
// evaluated yet.
TensorRef<Tensor<float, 3>> ref = ((t1 + t2) * 0.2f).exp();

// Use "ref" to access individual elements.  The expression is evaluated
// on the fly.
float at_0 = ref(0, 0, 0);
std::cout << ref(0, 1, 0);
```

Only use `TensorRef` when you need a subset of the values of the expression.
`TensorRef` only computes the values you access.
However note that if you are going to access all the values it will be much
 faster to materialize the results in a `Tensor` first.

In some cases, if the full `Tensor` result would be very large, you may save
memory by accessing it as a `TensorRef`.
But not always.
So don't count on it.


### Controlling How Expressions Are Evaluated

The tensor library provides several implementations of the various operations
such as contractions and convolutions.  The implementations are optimized for
different environments: single threaded on CPU, multi threaded on CPU, or on a
GPU using CUDA/HIP/SYCL.

You can choose which implementation to use with the `device()` call.  If
you do not choose an implementation explicitly the default implementation that
uses a single thread on the CPU is used.

The default implementation has been optimized for modern CPUs, taking
advantage of SSE, AVX, AVX-512, ARM NEON, SVE, RISC-V Vector (RVV), and other
SIMD instruction sets. Note that you need to pass compiler-dependent flags
to enable the use of these instructions (e.g. `-mavx2`, `-march=native`).

For example, the following code adds two tensors using the default
single-threaded CPU implementation:

```cpp
Tensor<float, 2> a(30, 40);
Tensor<float, 2> b(30, 40);
Tensor<float, 2> c = a + b;
```

To choose a different implementation you have to insert a `device()` call
before the assignment of the result.  For technical C++ reasons this requires
that the `Tensor` for the result be declared on its own.
This means that you have to know the size of the result.

```cpp
Eigen::Tensor<float, 2> c(30, 40);
c.device(...) = a + b;
```

The call to `device()` must be the last call on the left of the operator=.

You must pass to the `device()` call an Eigen device object.  There are
presently four devices you can use: `DefaultDevice`, `ThreadPoolDevice`,
`GpuDevice`, and `SyclDevice`.


#### Evaluating With the DefaultDevice

This is exactly the same as not inserting a `device()` call.

```cpp
DefaultDevice my_device;
c.device(my_device) = a + b;
```

#### Evaluating with a Thread Pool

To use `ThreadPoolDevice`, you must define `EIGEN_USE_THREADS` before
including the Tensor header:

```cpp
#define EIGEN_USE_THREADS
#include <unsupported/Eigen/Tensor>

// Create the Eigen ThreadPool.
Eigen::ThreadPool pool(8 /* number of threads in pool */);

// Create the Eigen ThreadPoolDevice.
Eigen::ThreadPoolDevice my_device(&pool, 4 /* number of threads to use */);

// Now just use the device when evaluating expressions.
Eigen::Tensor<float, 2> c(30, 50);
c.device(my_device) = a.contract(b, dot_product_dims);
```


#### Evaluating On GPU

To use `GpuDevice`, you must define `EIGEN_USE_GPU` before including the
Tensor header.  GPU tensors require explicitly allocating device memory
with CUDA or HIP APIs.

```cpp
#define EIGEN_USE_GPU
#include <unsupported/Eigen/Tensor>

// Allocate data on GPU.
float* d_a;
float* d_b;
float* d_c;
cudaMalloc((void**)&d_a, 30 * 40 * sizeof(float));
cudaMalloc((void**)&d_b, 30 * 40 * sizeof(float));
cudaMalloc((void**)&d_c, 30 * 40 * sizeof(float));

// Copy host data to device.
cudaMemcpy(d_a, h_a, 30 * 40 * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, 30 * 40 * sizeof(float), cudaMemcpyHostToDevice);

// Create device maps.
Eigen::TensorMap<Eigen::Tensor<float, 2>> gpu_a(d_a, 30, 40);
Eigen::TensorMap<Eigen::Tensor<float, 2>> gpu_b(d_b, 30, 40);
Eigen::TensorMap<Eigen::Tensor<float, 2>> gpu_c(d_c, 30, 40);

// Create a GPU device and evaluate.
Eigen::GpuStreamDevice stream;
Eigen::GpuDevice gpu_device(&stream);
gpu_c.device(gpu_device) = gpu_a + gpu_b;

// Synchronize and copy back.
cudaStreamSynchronize(stream.stream());
cudaMemcpy(h_c, d_c, 30 * 40 * sizeof(float), cudaMemcpyDeviceToHost);

cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
```

For HIP, replace `cuda*` calls with the corresponding `hip*` calls.

#### Asynchronous Device Execution

You can pass a callback to the `device()` call that will be invoked when the
computation completes.  This is supported by `ThreadPoolDevice` and `GpuDevice`.

```cpp
Eigen::Tensor<float, 2> c(30, 40);
auto done = []() { std::cout << "Computation complete!\n"; };
c.device(my_device, done) = a + b;
// The callback will be invoked when evaluation finishes.
```


## API Reference

### Datatypes

In the documentation of the tensor methods and Operation we mention datatypes
that are tensor-type specific:

#### \<Tensor-Type\>::Dimensions

Acts like an array of `Index`. Has a `size()` method (inherited from
`std::array`) and a static `count` member equal to the rank. Can be
indexed like an array to access individual values.  Used to represent the
dimensions of a tensor.  See `dimensions()`.

#### \<Tensor-Type\>::Index

Acts like an `int`.  Used for indexing tensors along their dimensions.  See
`operator()`, `dimension()`, and `size()`.

#### \<Tensor-Type\>::Scalar

Represents the datatype of individual tensor elements.  For example, for a
`Tensor<float>`, `Scalar` is the type `float`.  See `setConstant()`.

#### (Operation)

We use this pseudo type to indicate that a tensor Operation is returned by a
method.  We indicate in the text the type and dimensions of the tensor that the
Operation returns after evaluation.

The Operation will have to be evaluated, for example by assigning it to a
`Tensor`, before you can access the values of the resulting tensor.  You can also
access the values through a `TensorRef`.


## Built-in Tensor Methods

These are usual C++ methods that act on tensors immediately.  They are not
Operations which provide delayed evaluation of their results.  Unless specified
otherwise, all the methods listed below are available on all tensor classes:
`Tensor`, `TensorFixedSize`, and `TensorMap`.

## Metadata

### int NumDimensions

Constant value indicating the number of dimensions of a `Tensor`.
This is also known as the tensor rank.

```cpp
Eigen::Tensor<float, 2> a(3, 4);
std::cout << "Dims " << a.NumDimensions;
// Dims 2
```

### Dimensions dimensions()

Returns an array-like object representing the dimensions of the tensor.
The actual type of the `dimensions()` result is `<Tensor-Type>::Dimensions`.

```cpp
Eigen::Tensor<float, 2> a(3, 4);
const Eigen::Tensor<float, 2>::Dimensions& d = a.dimensions();
std::cout << "Dim size: " << d.size() << ", dim 0: " << d[0]
          << ", dim 1: " << d[1];
//  Dim size: 2, dim 0: 3, dim 1: 4
```

You can use `auto` to simplify the code:

```cpp
const auto& d = a.dimensions();
std::cout << "Dim size: " << d.size() << ", dim 0: " << d[0]
        << ", dim 1: " << d[1];
// Dim size: 2, dim 0: 3, dim 1: 4
```

### Index dimension(Index n)

Returns the n-th dimension of the tensor.  The actual type of the
`dimension()` result is `<Tensor-Type>::Index`, but you can
always use it like an int.

```cpp
Eigen::Tensor<float, 2> a(3, 4);
int dim1 = a.dimension(1);
std::cout << "Dim 1: " << dim1;
// Dim 1: 4
```

### Index size()

Returns the total number of elements in the tensor.  This is the product of all
the tensor dimensions.  The actual type of the `size()` result is
`<Tensor-Type>::Index`, but you can always use it like an int.

```cpp
Eigen::Tensor<float, 2> a(3, 4);
std::cout << "Size: " << a.size();
/// Size: 12
```

### Getting Dimensions From An Operation

A few operations provide `dimensions()` directly,
e.g. `TensorSlicingOp`.  Most operations defer calculating dimensions
until the operation is being evaluated.  If you need access to the dimensions
of a deferred operation, you can wrap it in a `TensorRef` (see
**Assigning to a TensorRef** above), which provides
`dimensions()` and `dimension()` as above.

`TensorRef` can also wrap the plain `Tensor` types, so this is a useful idiom in
templated contexts where the underlying object could be either a raw `Tensor`
or some deferred operation (e.g. a slice of a `Tensor`).  In this case, the
template code can wrap the object in a TensorRef and reason about its
dimensionality while remaining agnostic to the underlying type.


## Constructors

### Tensor

Creates a tensor of the specified size. The number of arguments must be equal
to the rank of the tensor. The content of the tensor is not initialized.

```cpp
Eigen::Tensor<float, 2> a(3, 4);
std::cout << "NumRows: " << a.dimension(0) << " NumCols: " << a.dimension(1) << endl;
// NumRows: 3 NumCols: 4
```
### TensorFixedSize

Creates a tensor of the specified size. The number of arguments in the `Sizes<>`
template parameter determines the rank of the tensor. The content of the tensor
is not initialized.

```cpp
Eigen::TensorFixedSize<float, Sizes<3, 4>> a;
std::cout << "Rank: " << a.rank() << endl;
// Rank: 2
std::cout << "NumRows: " << a.dimension(0)
          << " NumCols: " << a.dimension(1) << endl;
// NumRows: 3 NumCols: 4
```

### TensorMap

Creates a tensor mapping an existing array of data. The data must not be freed
until the `TensorMap` is discarded, and the size of the data must be large enough
to accommodate the coefficients of the tensor.

```cpp
float data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
Eigen::TensorMap<Tensor<float, 2>> a(data, 3, 4);
std::cout << "NumRows: " << a.dimension(0) << " NumCols: " << a.dimension(1) << endl;
// NumRows: 3 NumCols: 4
std::cout << "a(1, 2): " << a(1, 2) << endl;
// a(1, 2): 7
```

## Contents Initialization

When a new `Tensor` or a new `TensorFixedSize` are created, memory is allocated to
hold all the tensor elements, but the memory is not initialized.  Similarly,
when a new `TensorMap` is created on top of non-initialized memory the memory its
contents are not initialized.

You can use one of the methods below to initialize the tensor memory.  These
have an immediate effect on the tensor and return the tensor itself as a
result.  These are not tensor Operations which delay evaluation.

### \<Tensor-Type\> setConstant(const Scalar& val)

Sets all elements of the tensor to the constant value `val`.  `Scalar`
is the type of data stored in the tensor.  You can pass any value that is
convertible to that type.

Returns the tensor itself in case you want to chain another call.

```cpp
a.setConstant(12.3f);
std::cout << "Constant: " << endl << a << endl << endl;

// Constant:
// 12.3 12.3 12.3 12.3
// 12.3 12.3 12.3 12.3
// 12.3 12.3 12.3 12.3
```
Note that `setConstant()` can be used on any tensor where the element type
has a copy constructor and an `operator=()`:

```cpp
Eigen::Tensor<string, 2> a(2, 3);
a.setConstant("yolo");
std::cout << "String tensor: " << endl << a << endl << endl;

// String tensor:
// yolo yolo yolo
// yolo yolo yolo
```

### \<Tensor-Type\> setZero()

Fills the tensor with zeros.  Equivalent to `setConstant(Scalar(0))`.
Returns the tensor itself in case you want to chain another call.

```cpp
a.setZero();
std::cout << "Zeros: " << endl << a << endl << endl;

// Zeros:
// 0 0 0 0
// 0 0 0 0
// 0 0 0 0
```

### \<Tensor-Type\> setValues({..initializer_list})

Fills the tensor with explicit values specified in a std::initializer_list.
The type of the initializer list depends on the type and rank of the tensor.

If the tensor has rank N, the initializer list must be nested N times.  The
most deeply nested lists must contains P scalars of the `Tensor` type where P is
the size of the last dimension of the Tensor.

For example, for a `TensorFixedSize<float, Sizes<2, 3>>` the initializer list
must contains 2 lists of 3 floats each.

`setValues()` returns the tensor itself in case you want to chain another
call.

```cpp
Eigen::Tensor<float, 2> a(2, 3);
a.setValues({{0.0f, 1.0f, 2.0f}, {3.0f, 4.0f, 5.0f}});
std::cout << "a" << endl << a << endl << endl;

// a
// 0 1 2
// 3 4 5
```

If a list is too short, the corresponding elements of the tensor will not be
changed.  This is valid at each level of nesting.  For example the following
code only sets the values of the first row of the tensor.

```cpp
Eigen::Tensor<int, 2> a(2, 3);
a.setConstant(1000);
a.setValues({{10, 20, 30}});
std::cout << "a" << endl << a << endl << endl;
// a
// 10   20   30
// 1000 1000 1000
```

### \<Tensor-Type\> setRandom()

Fills the tensor with random values.  Returns the tensor itself in case you
want to chain another call.

```cpp
a.setRandom();
std::cout << "Random: " << endl << a << endl << endl;
// Random:
//   0.680375    0.59688  -0.329554    0.10794
//  -0.211234   0.823295   0.536459 -0.0452059
//   0.566198  -0.604897  -0.444451   0.257742
```

You can customize `setRandom()` by providing your own random number
generator as a template argument:

```cpp
a.setRandom<MyRandomGenerator>();
```

Here, `MyRandomGenerator` must be a struct with the following member
functions, where Scalar and Index are the same as `<Tensor-Type>::Scalar`
and `<Tensor-Type>::Index`.

See `struct UniformRandomGenerator` in TensorFunctors.h for an example.

```cpp
// Custom number generator for use with setRandom().
struct MyRandomGenerator {
  // Default and copy constructors. Both are needed
  MyRandomGenerator() { }
  MyRandomGenerator(const MyRandomGenerator& ) { }

  // Return a random value to be used.  "element_location" is the
  // location of the entry to set in the tensor, it can typically
  // be ignored.
  Scalar operator()(Eigen::DenseIndex element_location,
                    Eigen::DenseIndex /*unused*/ = 0) const {
    return <randomly generated value of type T>;
  }

  // Same as above but generates several numbers at a time.
  typename internal::packet_traits<Scalar>::type packetOp(
      Eigen::DenseIndex packet_location, Eigen::DenseIndex /*unused*/ = 0) const {
    return <a packet of randomly generated values>;
  }
};
```

You can also use one of the 2 random number generators that are part of the
tensor library:
*   UniformRandomGenerator
*   NormalRandomGenerator

## Data Access

The Tensor, TensorFixedSize, and TensorRef classes provide the following
accessors to access the tensor coefficients:

```cpp
const Scalar& operator()(const array<Index, NumIndices>& indices)
const Scalar& operator()(Index firstIndex, IndexTypes... otherIndices)
Scalar& operator()(const array<Index, NumIndices>& indices)
Scalar& operator()(Index firstIndex, IndexTypes... otherIndices)
```

The number of indices must be equal to the rank of the tensor. Moreover, these
accessors are not available on tensor expressions. In order to access the
values of a tensor expression, the expression must either be evaluated or
wrapped in a TensorRef.

### Scalar* data() and const Scalar* data() const

Returns a pointer to the storage for the tensor.  The pointer is const if the
tensor was const.  This allows direct access to the data.  The layout of the
data depends on the tensor layout: `RowMajor` or `ColMajor`.

This access is usually only needed for special cases, for example when mixing
Eigen Tensor code with other libraries.

Scalar is the type of data stored in the tensor.

```cpp
Eigen::Tensor<float, 2> a(3, 4);
float* a_data = a.data();
a_data[0] = 123.45f;
std::cout << "a(0, 0): " << a(0, 0);
// a(0, 0): 123.45
```

## Tensor Operations

All the methods documented below return non evaluated tensor `Operations`.
These can be chained: you can apply another `Tensor` Operation to the value
returned by the method.

The chain of Operation is evaluated lazily, typically when it is assigned to a
tensor.  See **Controlling When Expressions are Evaluated** for more details about
their evaluation.

### (Operation) constant(const Scalar& val)

Returns a tensor of the same type and dimensions as the original tensor but
where all elements have the value `val`.

This is useful, for example, when you want to add or subtract a constant from a
tensor, or multiply every element of a tensor by a scalar.
However, such operations can also be performed using operator overloads (see `operator+`).


```cpp
Eigen::Tensor<float, 2> a(2, 3);
a.setConstant(1.0f);
Eigen::Tensor<float, 2> b = a + a.constant(2.0f);
Eigen::Tensor<float, 2> c = b * b.constant(0.2f);
std::cout << "a" << endl << a << endl << endl;
std::cout << "b" << endl << b << endl << endl;
std::cout << "c" << endl << c << endl << endl;
// a
// 1 1 1
// 1 1 1

// b
// 3 3 3
// 3 3 3

// c
// 0.6 0.6 0.6
// 0.6 0.6 0.6
```

### (Operation) random()

Returns a tensor of the same type and dimensions as the current tensor
but where all elements have random values.

This is for example useful to add random values to an existing tensor.
The generation of random values can be customized in the same manner
as for `setRandom()`.

```cpp
Eigen::Tensor<float, 2> a(2, 3);
a.setConstant(1.0f);
Eigen::Tensor<float, 2> b = a + a.random();
std::cout << "a\n" << a << "\n\n";
std::cout << "b\n" << b << "\n\n";

// a
// 1 1 1
// 1 1 1
// b
// 1.68038   1.5662  1.82329
// 0.788766  1.59688
```

## Unary Element Wise Operations

All these operations take a single input tensor as argument and return a tensor
of the same type and dimensions as the tensor to which they are applied.  The
requested operations are applied to each element independently.

### (Operation) operator-()

Returns a tensor of the same type and dimensions as the original tensor
containing the opposite values of the original tensor.

```cpp
Eigen::Tensor<float, 2> a(2, 3);
a.setConstant(1.0f);
Eigen::Tensor<float, 2> b = -a;
std::cout << "a\n" << a << "\n\n";
std::cout << "b\n" << b << "\n\n";

// a
// 1 1 1
// 1 1 1
//
// b
// -1 -1 -1
// -1 -1 -1
```

### (Operation) sqrt()

Returns a tensor containing the square roots of the original tensor.

### (Operation) rsqrt()

Returns a tensor containing the inverse square roots (1/sqrt(x)) of the original tensor.

### (Operation) square()

Returns a tensor containing the squares of the original tensor values.

### (Operation) cube()

Returns a tensor containing the cubes (x^3) of the original tensor values.

### (Operation) inverse()

Returns a tensor containing the inverse (1/x) of the original tensor values.

### (Operation) exp()

Returns a tensor containing the exponential of the original tensor.

### (Operation) expm1()

Returns a tensor containing `exp(x) - 1` for each element. More accurate
than `exp(x) - 1` for small values of x.

### (Operation) log()

Returns a tensor containing the natural logarithms of the original tensor.

### (Operation) log1p()

Returns a tensor containing `log(1 + x)` for each element. More accurate
than `log(1 + x)` for small values of x.

### (Operation) log2()

Returns a tensor containing the base-2 logarithms of the original tensor.

### (Operation) abs()

Returns a tensor containing the absolute values of the original tensor.

### (Operation) sign()

Returns a tensor containing the sign (-1, 0, or +1) of each element.

### (Operation) arg()

Returns a tensor containing the complex argument (phase angle) of the
values of the original tensor.

### (Operation) real()

Returns a tensor containing the real part of the complex values of the
original tensor.  The result has a real-valued scalar type.

### (Operation) imag()

Returns a tensor containing the imaginary part of the complex values of
the original tensor. The result has a real-valued scalar type.

### (Operation) conjugate()

Returns a tensor containing the complex conjugate of each element.
For real-valued tensors, this is a no-op.

### (Operation) pow(Scalar exponent)

Returns a tensor containing the coefficients of the original tensor raised
to the power of the exponent.

The type of the exponent, Scalar, is always the same as the type of the
tensor coefficients.  For example, only integer exponents can be used in
conjunction with tensors of integer values.

You can use `cast()` to lift this restriction.  For example this computes
cubic roots of an int Tensor:

```cpp
Eigen::Tensor<int, 2> a(2, 3);
a.setValues({{0, 1, 8}, {27, 64, 125}});
Eigen::Tensor<double, 2> b = a.cast<double>().pow(1.0 / 3.0);
std::cout << "a" << endl << a << endl << endl;
std::cout << "b" << endl << b << endl << endl;

// a
// 0   1   8
// 27  64 125
//
// b
// 0 1 2
// 3 4 5
```

### (Operation) clip(Scalar min_val, Scalar max_val)

Returns a tensor with each element clamped to the range `[min_val, max_val]`.

```cpp
Eigen::Tensor<float, 1> a(5);
a.setValues({-2.0f, -0.5f, 0.0f, 0.5f, 2.0f});
Eigen::Tensor<float, 1> b = a.clip(-1.0f, 1.0f);
// b: -1 -0.5  0  0.5  1
```

### Rounding Operations

### (Operation) round()

Returns a tensor with each element rounded to the nearest integer.

### (Operation) rint()

Returns a tensor with each element rounded to the nearest integer
(using the current rounding mode).

### (Operation) ceil()

Returns a tensor with each element rounded up to the nearest integer.

### (Operation) floor()

Returns a tensor with each element rounded down to the nearest integer.

### Predicates

### (Operation) (isnan)()

Returns a bool tensor indicating which elements are NaN.

```cpp
Eigen::Tensor<float, 1> a(3);
a.setValues({1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f});
Eigen::Tensor<bool, 1> b = a.isnan().cast<bool>();
// b: false true false
```

### (Operation) (isinf)()

Returns a bool tensor indicating which elements are infinite.

### (Operation) (isfinite)()

Returns a bool tensor indicating which elements are finite (not NaN or Inf).

### Hyperbolic and Activation Functions

### (Operation) tanh()

Returns a tensor containing the hyperbolic tangent of each element.

### (Operation) sigmoid()

Returns a tensor containing the logistic sigmoid (1/(1+exp(-x))) of each element.

### Error Functions

### (Operation) erf()

Returns a tensor containing the error function of each element.

### (Operation) erfc()

Returns a tensor containing the complementary error function (1 - erf(x)) of each element.

### (Operation) ndtri()

Returns a tensor containing the inverse of the normal cumulative distribution function of each element.

### Special Math Functions

These require including `<unsupported/Eigen/SpecialFunctions>` in addition to
the Tensor header.

### (Operation) lgamma()

Returns a tensor containing the log-gamma function of each element.

### (Operation) digamma()

Returns a tensor containing the digamma (psi) function of each element.

### (Operation) bessel_i0(), bessel_i0e(), bessel_i1(), bessel_i1e()

Modified Bessel functions of the first kind. The `e` variants are exponentially scaled.

### (Operation) bessel_j0(), bessel_j1()

Bessel functions of the first kind.

### (Operation) bessel_y0(), bessel_y1()

Bessel functions of the second kind.

### (Operation) bessel_k0(), bessel_k0e(), bessel_k1(), bessel_k1e()

Modified Bessel functions of the second kind. The `e` variants are exponentially scaled.

### (Operation) igamma(const OtherDerived& other)

Regularized lower incomplete gamma function. `this` is the parameter `a` and
`other` is `x`.

### (Operation) igammac(const OtherDerived& other)

Regularized upper incomplete gamma function (1 - igamma).

### (Operation) zeta(const OtherDerived& other)

Riemann zeta function. `this` is `x` and `other` is `q`.

### (Operation) polygamma(const OtherDerived& other)

Polygamma function. `this` is `n` and `other` is `x`.

### Scalar Arithmetic

### (Operation) operator*(Scalar s)

Multiplies every element of the input tensor by the scalar `s`:
```cpp
Eigen::Tensor<int, 2> a(2, 3);
a.setValues({{1, 2, 3},
                {4, 5, 6}});
Eigen::Tensor<int,2> scaled_a = a * 2;

std::cout << "a\n" << a << "\n";
std::cout << "scaled_a\n" << scaled_a << "\n";

// a
// 1 2 3
// 4 5 6
//
// scaled_a
// 2  4  6
// 8 10 12
```
### (Operation) operator+ (Scalar s)
Adds `s` to every element in the tensor.

### (Operation) operator- (Scalar s)
Subtracts `s` from every element in the tensor.

### (Operation) operator/ (Scalar s)
Divides every element in the tensor by `s`.

### (Operation) operator% (Scalar s)
Computes the element-wise modulus (remainder) of each tensor element divided by `s`.

**Only integer types are supported.**
For floating-point tensors, implement a `unaryExpr` using `std::fmod`.

### (Operation)  cwiseMax(Scalar threshold)
Returns a tensor where each element is the maximum of the original element and the
scalar threshold.
```cpp
Eigen::Tensor<int, 2> a(2, 3);
a.setValues({{0, 100, 200}, {300, 400, 500}});

Eigen::Tensor<int, 2> b(2, 3);
b.setValues({{-1, -2, 300}, {-4, 555, -6}});

Eigen::Tensor<int, 2> c = a.cwiseMax(b);

std::cout << "a\n" << a << "\n"
            << "b\n" << b << "\n"
            << "c\n" << c << "\n";

// a
//   0 100 200
// 300 400 500

// b
// -1  -2 300
// -4 555  -6

// c
//   0 100 300
// 300 555 500
```
### (Operation)  cwiseMin(Scalar threshold)
Returns a tensor where each element is the minimum of the original element and the
scalar threshold.

```cpp
Eigen::Tensor<int, 2> a(2, 2);
a.setValues({{0, 100}, {300, -900}});

Eigen::Tensor<int, 2> b(2, 2);
b.setValues({{-1, -2}, {400, 555}});

Eigen::Tensor<int, 2> c = a.cwiseMin(b);

std::cout << "a\n" << a << "\n"
          << "b\n" << b << "\n"
          << "c\n" << c << "\n";

// a
//   0  100
// 300 -900

// b
//  -1  -2
// 400 555

// c
//  -1   -2
// 300 -900
```

### NaN Propagation for cwiseMax and cwiseMin

The `cwiseMax` and `cwiseMin` operations accept an optional template parameter
controlling NaN propagation:

*   `cwiseMax<Eigen::PropagateNaN>(other)` — if either operand is NaN, the result is NaN.
*   `cwiseMax<Eigen::PropagateNumbers>(other)` — NaN is treated as missing; the non-NaN value wins.
*   `cwiseMax(other)` — default behavior (fast; may or may not propagate NaN, depends on platform).

```cpp
Eigen::Tensor<float, 1> a(3), b(3);
a.setValues({1.0f, NAN, 3.0f});
b.setValues({2.0f, 2.0f, NAN});

Eigen::Tensor<float, 1> c = a.cwiseMax<Eigen::PropagateNaN>(b);
// c: 2.0, NaN, NaN

Eigen::Tensor<float, 1> d = a.cwiseMax<Eigen::PropagateNumbers>(b);
// d: 2.0, 2.0, 3.0
```


### (Operation)  unaryExpr(const CustomUnaryOp& func)
Applies a user defined function to each element in the tensor.
Supports lambdas or functor structs with an operator().

Using lambda:
```cpp
Eigen::Tensor<float, 2> a(2, 3);
a.setValues({{0, -.5, -1}, {.5, 1.5, 2.0}});
auto my_func = [](float el){ return std::abs(el + 0.5f);};
Eigen::Tensor<float, 2> b = a.unaryExpr(my_func);
std::cout << "a\n" << a << "\n"
        << "b\n" << b << "\n";
=>
a
    0  -0.5   -1
0.5   1.5    2
b
0.5     0  0.5
    1     2  2.5
```

Using a functor to normalize and clamp values to `[-1.0, 1.0]`:

```cpp
template<typename Scalar>
struct NormalizedClamp {
NormalizedClamp(Scalar lo, Scalar hi) : _lo(lo), _hi(hi) {}
Scalar operator()(Scalar x) const {
    if (x < _lo) return Scalar(0);
    if (x > _hi) return Scalar(1);
    return (x - _lo) / (_hi - _lo);
}
Scalar _lo, _hi;
};

Eigen::Tensor<float, 2> c = a.unaryExpr(NormalizedClamp<float>(-1.0f, 1.0f));
std::cout << "c\n" << c << "\n";

// c
// 0.5    0.25    0
// 0.75   1       1
```

### Bitwise and Boolean Unary Operations

### (Operation) operator~()

Bitwise NOT of each element (integer types only).

### (Operation) operator!()

Boolean NOT of each element.


## Binary Element Wise Operations

These operations take two input tensors as arguments. The 2 input tensors should
be of the same type and dimensions. The result is a tensor of the same
dimensions as the tensors to which they are applied, and unless otherwise
specified it is also of the same type. The requested operations are applied to
each pair of elements independently.

### (Operation) operator+(const OtherDerived& other)

Returns a tensor of the same type and dimensions as the input tensors
containing the coefficient wise sums of the inputs.

### (Operation) operator-(const OtherDerived& other)

Returns a tensor of the same type and dimensions as the input tensors
containing the coefficient wise differences of the inputs.

### (Operation) operator*(const OtherDerived& other)

Returns a tensor of the same type and dimensions as the input tensors
containing the coefficient wise products of the inputs.

### (Operation) operator/(const OtherDerived& other)

Returns a tensor of the same type and dimensions as the input tensors
containing the coefficient wise quotients of the inputs.

This operator is not supported for integer types.

### (Operation) cwiseMax(const OtherDerived& other)

Returns a tensor of the same type and dimensions as the input tensors
containing the coefficient wise maximums of the inputs.

### (Operation) cwiseMin(const OtherDerived& other)

Returns a tensor of the same type and dimensions as the input tensors
containing the coefficient wise minimums of the inputs.

### (Operation) binaryExpr(const OtherDerived& other, const CustomBinaryOp& func)

Applies a custom binary functor element-wise to two tensors.

```cpp
Eigen::Tensor<float, 2> a(2, 3), b(2, 3);
a.setRandom(); b.setRandom();
auto my_op = [](float x, float y) { return x * x + y * y; };
Eigen::Tensor<float, 2> c = a.binaryExpr(b, my_op);
```

### (Operation) Logical operators

The following boolean operators are supported:

 * `operator&&(const OtherDerived& other)`
 * `operator||(const OtherDerived& other)`
 * `operator<(const OtherDerived& other)`
 * `operator<=(const OtherDerived& other)`
 * `operator>(const OtherDerived& other)`
 * `operator>=(const OtherDerived& other)`
 * `operator==(const OtherDerived& other)`
 * `operator!=(const OtherDerived& other)`

 as well as bitwise operators:

 * `operator&(const OtherDerived& other)`
 * `operator|(const OtherDerived& other)`
 * `operator^(const OtherDerived& other)`

The resulting tensor retains the input scalar type.

Scalar comparison variants are also available (e.g. `a < 0.5f`).

## Selection (select(const ThenDerived& thenTensor, const ElseDerived& elseTensor)

Selection is a coefficient-wise ternary operator that is the tensor equivalent
to the if-then-else operation.

```cpp
    Tensor<bool, 3> if = ...;
    Tensor<float, 3> then = ...;
    Tensor<float, 3> else = ...;
    Tensor<float, 3> result = if.select(then, else);
```

The 3 arguments must be of the same dimensions, which will also be the dimension
of the result.  The 'if' tensor must be of type boolean, the 'then' and the
'else' tensor must be of the same type, which will also be the type of the
result.

Each coefficient in the result is equal to the corresponding coefficient in the
'then' tensor if the corresponding value in the 'if' tensor is true. If not, the
resulting coefficient will come from the 'else' tensor.


## Contraction

Tensor *contractions* are a generalization of the matrix product to the
multidimensional case.

```cpp
// Create 2 matrices using tensors of rank 2
Eigen::Tensor<int, 2> a(2, 3);
a.setValues({{1, 2, 3}, {6, 5, 4}});
Eigen::Tensor<int, 2> b(3, 2);
b.setValues({{1, 2}, {4, 5}, {5, 6}});

// Compute the traditional matrix product
Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
Eigen::Tensor<int, 2> AB = a.contract(b, product_dims);

// Compute the product of the transpose of the matrices
Eigen::array<Eigen::IndexPair<int>, 1> transposed_product_dims = { Eigen::IndexPair<int>(0, 1) };
Eigen::Tensor<int, 2> AtBt = a.contract(b, transposed_product_dims);

// Contraction to scalar value using a double contraction.
// First coordinate of both tensors are contracted as well as both second coordinates, i.e., this computes the sum of the squares of the elements.
Eigen::array<Eigen::IndexPair<int>, 2> double_contraction_product_dims = { Eigen::IndexPair<int>(0, 0), Eigen::IndexPair<int>(1, 1) };
Eigen::Tensor<int, 0> AdoubleContractedA = a.contract(a, double_contraction_product_dims);

// Extracting the scalar value of the tensor contraction for further usage
int value = AdoubleContractedA(0);
```

## Reduction Operations

A *Reduction* operation returns a tensor with fewer dimensions than the
original tensor.  The values in the returned tensor are computed by applying a
*reduction operator* to slices of values from the original tensor.  You specify
the dimensions along which the slices are made.

The Eigen Tensor library provides a set of predefined reduction operators such
as `maximum()` and `sum()` and lets you define additional operators by
implementing a few methods from a reductor template.

### Reduction Dimensions

All reduction operations take a single parameter of type
`<TensorType>::``Dimensions` which can always be specified as an array of
ints.  These are called the "reduction dimensions."  The values are the indices
of the dimensions of the input tensor over which the reduction is done.  The
parameter can have at most as many element as the rank of the input tensor;
each element must be less than the tensor rank, as it indicates one of the
dimensions to reduce.

Each dimension of the input tensor should occur at most once in the reduction
dimensions as the implementation does not remove duplicates.

The order of the values in the reduction dimensions does not affect the
results, but the code may execute faster if you list the dimensions in
increasing order.

Example: Reduction along one dimension.
```cpp
// Create a tensor of 2 dimensions
Eigen::Tensor<int, 2> a(2, 3);
a.setValues({{1, 2, 3}, {6, 5, 4}});
// Reduce it along the second dimension (1)...
Eigen::array<int, 1> dims{1 /* dimension to reduce */};
// ...using the "maximum" operator.
// The result is a tensor with one dimension.  The size of
// that dimension is the same as the first (non-reduced) dimension of a.
Eigen::Tensor<int, 1> b = a.maximum(dims);
std::cout << "a" << endl << a << endl << endl;
std::cout << "b" << endl << b << endl << endl;

// a
// 1 2 3
// 6 5 4

// b
// 3
// 6
```
Example: Reduction along two dimensions.
```cpp
Eigen::Tensor<float, 3, Eigen::ColMajor> a(2, 3, 4);
a.setValues({{{0.0f, 1.0f, 2.0f, 3.0f},
                {7.0f, 6.0f, 5.0f, 4.0f},
                {8.0f, 9.0f, 10.0f, 11.0f}},
                {{12.0f, 13.0f, 14.0f, 15.0f},
                {19.0f, 18.0f, 17.0f, 16.0f},
                {20.0f, 21.0f, 22.0f, 23.0f}}});
// The tensor a has 3 dimensions.  We reduce along the
// first 2, resulting in a tensor with a single dimension
// of size 4 (the last dimension of a.)
// Note that we pass the array of reduction dimensions
// directly to the maximum() call.
Eigen::Tensor<float, 1, Eigen::ColMajor> b =
    a.maximum(Eigen::array<int, 2>{0, 1});
std::cout << "b" << endl << b << endl << endl;

// b
// 20
// 21
// 22
// 23
```
#### Reduction along all dimensions

As a special case, if you pass no parameter to a reduction operation the
original tensor is reduced along *all* its dimensions.  The result is a
scalar, represented as a zero-dimension tensor.

```cpp
Eigen::Tensor<float, 3> a(2, 3, 4);
a.setValues({{{0.0f, 1.0f, 2.0f, 3.0f},
              {7.0f, 6.0f, 5.0f, 4.0f},
              {8.0f, 9.0f, 10.0f, 11.0f}},
              {{12.0f, 13.0f, 14.0f, 15.0f},
              {19.0f, 18.0f, 17.0f, 16.0f},
              {20.0f, 21.0f, 22.0f, 23.0f}}});
// Reduce along all dimensions using the sum() operator.
Eigen::Tensor<float, 0> b = a.sum();
std::cout << "b\n" << b;

// b
// 276
```
You can extract the scalar directly by casting the expression and extract the first and only coefficient:
```cpp
float sum = static_cast<Eigen::Tensor<float, 0>>(a.sum())();
```

### (Operation) sum(const Dimensions& reduction_dims)
### (Operation) sum()

Reduce a tensor using the `sum()` operator.  The resulting values
are the sum of the reduced values.

### (Operation) mean(const Dimensions& reduction_dims)
### (Operation) mean()

Reduce a tensor using the `mean()` operator.  The resulting values
are the mean of the reduced values.

### (Operation) maximum(const Dimensions& reduction_dims)
### (Operation) maximum()

Reduce a tensor using the `maximum()` operator.  The resulting values are the
largest of the reduced values.

### (Operation) minimum(const Dimensions& reduction_dims)
### (Operation) minimum()

Reduce a tensor using the `minimum()` operator.  The resulting values
are the smallest of the reduced values.

### NaN Propagation for maximum and minimum

Like `cwiseMax` and `cwiseMin`, the `maximum` and `minimum` reductions accept
an optional NaN propagation template parameter:

```cpp
// If any element along the reduction is NaN, the result is NaN.
Eigen::Tensor<float, 1> b = a.maximum<Eigen::PropagateNaN>(dims);

// NaN values are ignored during reduction.
Eigen::Tensor<float, 1> c = a.maximum<Eigen::PropagateNumbers>(dims);
```

### (Operation) prod(const Dimensions& reduction_dims)
### (Operation) prod()

Reduce a tensor using the `prod()` operator.  The resulting values
are the product of the reduced values.

### (Operation) all(const Dimensions& reduction_dims)
### (Operation) all()
Reduce a tensor using the `all()` operator.  Casts tensor to bool and then checks
whether all elements are true.  Runs through all elements rather than
short-circuiting, so may be significantly inefficient.

### (Operation) any(const Dimensions& reduction_dims)
### (Operation) any()
Reduce a tensor using the `any()` operator.  Casts tensor to bool and then checks
whether any element is true.  Runs through all elements rather than
short-circuiting, so may be significantly inefficient.


### (Operation) argmax(const Dimensions& reduction_dim)
### (Operation) argmax()

Reduce a tensor using the `argmax()` operator.

The resulting values are the indices of the largest elements along the specified dimension.

Only a single `reduction_dim` is supported.

If multiple elements share the maximum value, the one with the **lowest index** is returned.

```cpp
Eigen::Tensor<float, 2> a(2, 3);
a.setValues({{1, 4, 8}, {3, 4, 2}});

Eigen::Tensor<Eigen::Index, 1> argmax_dim0 = a.argmax(0);

std::cout << "a:\n" << a << "\n";
for (int i = 0; i < argmax_dim0.size(); ++i) {
    std::cout << "argmax along dim 0 at index " << i << " = " << argmax_dim0(i) << "\n";
}

// a:
// 1 4 8
// 3 4 2
// argmax along dim 0 at index 0 = 1
// argmax along dim 0 at index 1 = 0
// argmax along dim 0 at index 2 = 0
```

 To compute the index of the global maximum, use the overload without arguments (which flattens the tensor).


```cpp
Eigen::Tensor<Eigen::Index, 0> argmax_flat = a.argmax();
std::cout << "Flat argmax index: " << argmax_flat();

// Flat argmax index: 4
```

### (Operation) argmin(const Dimensions& reduction_dim)
### (Operation) argmin()
See `argmax`.

### (Operation) reduce(const Dimensions& reduction_dims, const Reducer& reducer)

Reduce a tensor using a user-defined reduction operator.  See `SumReducer`
in TensorFunctors.h for information on how to implement a reduction operator.


## Trace

A *Trace* operation returns a tensor with fewer dimensions than the original
tensor. It returns a tensor whose elements are the sum of the elements of the
original tensor along the main diagonal for a list of specified dimensions, the
"trace dimensions". Similar to the `Reduction Dimensions`, the trace dimensions
are passed as an input parameter to the operation, are of type `<TensorType>::``Dimensions`
, and have the same requirements when passed as an input parameter. In addition,
the trace dimensions must have the same size.

Example: Trace along 2 dimensions.

```cpp
// Create a tensor of 3 dimensions
Eigen::Tensor<int, 3> a(2, 2, 3);
a.setValues({{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}});
// Specify the dimensions along which the trace will be computed.
// In this example, the trace can only be computed along the dimensions
// with indices 0 and 1
Eigen::array<int, 2> dims{0, 1};
// The output tensor contains all but the trace dimensions.
Tensor<int, 1> a_trace = a.trace(dims);
std::cout << "a_trace:" << endl;
std::cout << a_trace << endl;

// a_trace:
// 11
// 13
// 15
```

### (Operation) trace(const Dimensions& new_dims)
### (Operation) trace()

As a special case, if no parameter is passed to the operation, trace is computed
along *all* dimensions of the input tensor.

Example: Trace along all dimensions.

```cpp
// Create a tensor of 3 dimensions, with all dimensions having the same size.
Eigen::Tensor<int, 3> a(3, 3, 3);
a.setValues({{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
            {{10, 11, 12}, {13, 14, 15}, {16, 17, 18}},
            {{19, 20, 21}, {22, 23, 24}, {25, 26, 27}}});
// Result is a zero dimension tensor
Tensor<int, 0> a_trace = a.trace();
std::cout<<"a_trace:"<<endl;
std::cout<<a_trace<<endl;

// a_trace:
// 42
```

## Scan Operations

A *Scan* operation returns a tensor with the same dimensions as the original
tensor. The operation performs an inclusive scan along the specified
axis, which means it computes a running total along the axis for a given
reduction operation.
If the reduction operation corresponds to summation, then this computes the
prefix sum of the tensor along the given axis.

Example:
Cumulative sum along the second dimension

```cpp
// Create a tensor of 2 dimensions
Eigen::Tensor<int, 2> a(2, 3);
a.setValues({{1, 2, 3}, {4, 5, 6}});
// Scan it along the second dimension (1) using summation
Eigen::Tensor<int, 2> b = a.cumsum(1);
// The result is a tensor with the same size as the input
std::cout << "a" << endl << a << endl << endl;
std::cout << "b" << endl << b << endl << endl;

// a
// 1 2 3
// 4 5 6

// b
// 1  3  6
// 4  9 15
```

### (Operation) cumsum(const Index& axis, bool exclusive = false)

Perform a scan by summing consecutive entries.

When `exclusive` is true, element `i` contains the sum of all elements before
index `i` (exclusive prefix sum). The first element along the axis is 0.

```cpp
Eigen::Tensor<int, 1> a(4);
a.setValues({1, 2, 3, 4});

Eigen::Tensor<int, 1> inclusive = a.cumsum(0);           // 1, 3, 6, 10
Eigen::Tensor<int, 1> exclusive = a.cumsum(0, true);     // 0, 1, 3, 6
```

### (Operation) cumprod(const Index& axis, bool exclusive = false)

Perform a scan by multiplying consecutive entries.

When `exclusive` is true, element `i` contains the product of all elements
before index `i`. The first element along the axis is 1.

```cpp
Eigen::Tensor<int, 1> a(4);
a.setValues({1, 2, 3, 4});

Eigen::Tensor<int, 1> inclusive = a.cumprod(0);           // 1, 2, 6, 24
Eigen::Tensor<int, 1> exclusive = a.cumprod(0, true);     // 1, 1, 2, 6
```

## Convolutions

### (Operation) convolve(const Kernel& kernel, const Dimensions& dims)

Returns a tensor that is the output of the convolution of the input tensor with the kernel,
along the specified dimensions of the input tensor. The dimension size for dimensions of the output tensor
which were part of the convolution will be reduced by the formula:
```cpp
output_dim_size = input_dim_size - kernel_dim_size + 1 // (requires: input_dim_size >= kernel_dim_size).
```
The dimension sizes for dimensions that were not part of the convolution will remain the same.
Performance of the convolution can depend on the length of the stride(s) of the input tensor dimension(s) along which the
convolution is computed (the first dimension has the shortest stride for `ColMajor`, whereas `RowMajor`'s shortest stride is
for the last dimension).

```cpp
// Compute convolution along the second and third dimension.
Tensor<float, 4, DataLayout> input(3, 3, 7, 11);
Tensor<float, 2, DataLayout> kernel(2, 2);
Tensor<float, 4, DataLayout> output(3, 2, 6, 11);
input.setRandom();
kernel.setRandom();

Eigen::array<ptrdiff_t, 2> dims{1, 2};  // Specify second and third dimension for convolution.
output = input.convolve(kernel, dims);

for (int i = 0; i < 3; ++i) {
  for (int j = 0; j < 2; ++j) {
    for (int k = 0; k < 6; ++k) {
      for (int l = 0; l < 11; ++l) {
        const float result = output(i,j,k,l);
        const float expected = input(i,j+0,k+0,l) * kernel(0,0) +
                               input(i,j+1,k+0,l) * kernel(1,0) +
                               input(i,j+0,k+1,l) * kernel(0,1) +
                               input(i,j+1,k+1,l) * kernel(1,1);
        VERIFY_IS_APPROX(result, expected);
      }
    }
  }
}
```

## FFT (Fast Fourier Transform)

### (Operation) fft\<FFTResultType, FFTDirection\>(const FFTDims& dims)

Computes the Fast Fourier Transform of the input tensor along the specified
dimensions.

Template parameters:

| Parameter | Values | Description |
|-----------|--------|-------------|
| `FFTResultType` | `RealPart`, `ImagPart`, `BothParts` | Which part(s) of the result to return |
| `FFTDirection` | `FFT_FORWARD`, `FFT_REVERSE` | Forward or inverse transform |

When `FFTResultType` is `BothParts`, the output scalar type is
`std::complex<Scalar>`. When `RealPart` or `ImagPart`, the output retains
the real scalar type.

```cpp
// Forward FFT of a 2D tensor along both dimensions.
Eigen::Tensor<float, 2> input(8, 16);
input.setRandom();
Eigen::array<int, 2> fft_dims{{0, 1}};

// Get the full complex result.
Eigen::Tensor<std::complex<float>, 2> complex_result =
    input.fft<Eigen::BothParts, Eigen::FFT_FORWARD>(fft_dims);

// Get only the real part.
Eigen::Tensor<float, 2> real_result =
    input.fft<Eigen::RealPart, Eigen::FFT_FORWARD>(fft_dims);

// Inverse FFT to recover the original signal.
Eigen::Tensor<float, 2> recovered =
    complex_result.fft<Eigen::RealPart, Eigen::FFT_REVERSE>(fft_dims);
```

The FFT uses the Cooley-Tukey algorithm for power-of-2 sizes and falls back to
the Bluestein algorithm for arbitrary sizes.

## Geometrical Operations

These operations return a `Tensor` with different dimensions than the original
`Tensor`.  They can be used to access slices of tensors, see them with different
dimensions, or pad tensors with additional data.

### (Operation) reshape(const Dimensions& new_dims)

Returns a view of the input tensor that has been reshaped to the specified
new dimensions.

The argument `new_dims` is an array of Index values.

The rank of the resulting tensor is equal to the number of elements in `new_dims`.

The product of all the sizes in the new dimension array must be equal to
the number of elements in the input tensor.

```cpp
// Increase the rank of the input tensor by introducing a new dimension
// of size 1.
Tensor<float, 2> input(7, 11);
array<int, 3> three_dims{{7, 11, 1}};
Tensor<float, 3> result = input.reshape(three_dims);

// Decrease the rank of the input tensor by merging 2 dimensions;
array<int, 1> one_dim{{7 * 11}};
Tensor<float, 1> result = input.reshape(one_dim);
```

This operation does not move any data in the input tensor, so the resulting
contents of a reshaped `Tensor` depend on the data layout of the original `Tensor`.

For example this is what happens when you `reshape()` a 2D `ColMajor` tensor
to one dimension:

```cpp
Eigen::Tensor<float, 2, Eigen::ColMajor> a(2, 3);
a.setValues({{0.0f, 100.0f, 200.0f}, {300.0f, 400.0f, 500.0f}});
Eigen::array<Eigen::DenseIndex, 1> one_dim{3 * 2};
Eigen::Tensor<float, 1, Eigen::ColMajor> b = a.reshape(one_dim);
std::cout << "b" << endl << b << endl;

// b
//   0
// 300
// 100
// 400
// 200
// 500
```

This is what happens when the 2D `Tensor` is `RowMajor`:

```cpp
Eigen::Tensor<float, 2, Eigen::RowMajor> a(2, 3);
a.setValues({{0.0f, 100.0f, 200.0f}, {300.0f, 400.0f, 500.0f}});
Eigen::array<Eigen::DenseIndex, 1> one_dim{3 * 2};
Eigen::Tensor<float, 1, Eigen::RowMajor> b = a.reshape(one_dim);
std::cout << "b" << endl << b << endl;

// b
//   0
// 100
// 200
// 300
// 400
// 500
```

The reshape operation is a lvalue. In other words, it can be used on the left
side of the assignment operator.

The previous example can be rewritten as follow:

```cpp
Eigen::Tensor<float, 2, Eigen::ColMajor> a(2, 3);
a.setValues({{0.0f, 100.0f, 200.0f}, {300.0f, 400.0f, 500.0f}});
Eigen::array<Eigen::DenseIndex, 2> two_dim{2, 3};
Eigen::Tensor<float, 1, Eigen::ColMajor> b(6);
b.reshape(two_dim) = a;
std::cout << "b" << endl << b << endl;

// b
//   0
// 300
// 100
// 400
// 200
// 500
```

Note that "b" itself was not reshaped but that instead the assignment is done to
the reshape view of b.

### (Operation) shuffle(const Shuffle& shuffle)

Returns a view of the input tensor whose dimensions have been
reordered according to the specified permutation.

The argument `shuffle` is an array of `Index` values:
* Its size is the rank of the input tensor.
* It must contain a permutation of `[0, 1, ..., rank - 1]`.
* The `i`-th dimension of the output tensor corresponds to the size of the dimension at position `shuffle[i]` in the input tensor. For example:

```cpp
// Shuffle all dimensions to the left by 1.
Tensor<float, 3> input(20, 30, 50);
// ... set some values in input.
Tensor<float, 3> output = input.shuffle({1, 2, 0});

eigen_assert(output.dimension(0) == 30);
eigen_assert(output.dimension(1) == 50);
eigen_assert(output.dimension(2) == 20);

// Indices into the output tensor are shuffled accordingly to formulate
// indices into the input tensor.
eigen_assert(output(3, 7, 11) == input(11, 3, 7));

// In general:
eigen_assert(output(..., indices[shuffle[i]], ...) ==
             input(..., indices[i], ...));
```

The shuffle operation results in a lvalue, which means that it can be assigned
to. In other words, it can be used on the left side of the assignment operator.

Let's rewrite the previous example to take advantage of this feature:

```cpp
// Shuffle all dimensions to the left by 1.
Tensor<float, 3> input(20, 30, 50);
input.setRandom();
Tensor<float, 3> output(30, 50, 20);
output.shuffle({2, 0, 1}) = input;
```

### (Operation) stride(const Strides& strides)

Returns a view of the input tensor that strides (skips stride-1
elements) along each of the dimensions.

The argument strides is an array of `Index` values:
* Its size is the rank of the input tensor.
* Must be >= 1

 The dimensions of the resulting tensor are `ceil(input_dimensions[i] / strides[i])`.

For example this is what happens when you `stride()` a 2D tensor:

```cpp
Eigen::Tensor<int, 2> a(4, 3);
a.setValues({{0, 100, 200},
             {300, 400, 500},
             {600, 700, 800},
             {900, 1000, 1100}});
Eigen::array<Eigen::DenseIndex, 2> strides{3, 2};
Eigen::Tensor<int, 2> b = a.stride(strides);
std::cout << "b" << endl << b << endl;
// b
//    0   200
//  900  1100
```

It is possible to assign a tensor to a stride:
```cpp
Tensor<float, 3> input(20, 30, 50);
input.setRandom();
Tensor<float, 3> output(40, 90, 200);
output.stride({2, 3, 4}) = input;
```

### (Operation) slice(const StartIndices& offsets, const Sizes& extents)

Returns a sub-tensor of the given tensor. For each dimension i, the slice is
made of the coefficients stored between `offset[i]` and `offset[i] + extents[i]` in
the input tensor.

```cpp
Eigen::Tensor<int, 2> a(4, 3);
a.setValues({{0, 100, 200}, {300, 400, 500},
             {600, 700, 800}, {900, 1000, 1100}});
Eigen::array<Eigen::Index, 2> offsets = {1, 0};
Eigen::array<Eigen::Index, 2> extents = {2, 2};
Eigen::Tensor<int, 2> slice = a.slice(offsets, extents);
std::cout << "a" << endl << a << endl;
// a
//    0   100   200
//  300   400   500
//  600   700   800
//  900  1000  1100

std::cout << "slice" << endl << slice << endl;
// slice
//  300   400
//  600   700
```

### (Operation) stridedSlice(const StartIndices& start, const StopIndices& stop, const Strides& strides)

Returns a sub-tensor by selecting elements using `start`, `stop` (exclusive), and `strides` for each dimension.

This is similar to slicing in Python using [start:stop:step].

``` cpp
Eigen::Tensor<int, 2> a(4, 6);
a.setValues({{  0,  10,  20,  30,  40,   50},
             {100, 110, 120, 130, 140,  150},
             {200, 210, 220, 230, 240,  250},
             {300, 310, 320, 330, 340,  350}});

Eigen::array<Eigen::Index, 2> start = {1, 1};
Eigen::array<Eigen::Index, 2> stop  = {4, 6}; // Stop is exclusive
Eigen::array<Eigen::Index, 2> strides = {2, 2};

Eigen::Tensor<int, 2> sub = a.stridedSlice(start, stop, strides);

std::cout << "a\n" << a << "\n";
std::cout << "sub\n" << sub << "\n";

// a
//   0  10  20  30  40  50
// 100 110 120 130 140 150
// 200 210 220 230 240 250
// 300 310 320 330 340 350

// sub
// 110 130 150
// 310 330 350
```
It is also possible to assign to a strided slice:

``` cpp
Eigen::Tensor<int, 2> b(sub.dimensions());
b.setConstant(-1);
a.stridedSlice(start, stop, strides) = b;
std::cout << "modified a\n" << a << "\n";


// modified a
//   0  10  20  30  40  50
// 100  -1 120  -1 140  -1
// 200 210 220 230 240 250
// 300  -1 320  -1 340  -1

```
### (Operation) chip(const Index offset, const Index dim)

A chip is a special kind of slice.
It is the subtensor at the given offset in the dimension `dim`.

The returned tensor has one fewer dimension than the input tensor: the dimension dim is removed.

For example, a matrix chip would be either a row or a column of the input matrix:

```cpp
Eigen::Tensor<int, 2> a(4, 3);
a.setValues({{0, 100, 200}, {300, 400, 500},
             {600, 700, 800}, {900, 1000, 1100}});
Eigen::Tensor<int, 1> row_3 = a.chip(2, 0);
Eigen::Tensor<int, 1> col_2 = a.chip(1, 1);
std::cout << "a\n" << a << "\n";

// a
//    0   100   200
//  300   400   500
//  600   700   800
//  900  1000  1100

std::cout << "row_3\n" << row_3 << "\n";
// row_3
//    600   700   800

std::cout << "col_2\n" << col_2 << "\n";
// col_2
//   100   400   700    1000
```

It is possible to assign values to a tensor chip since the chip operation is a
lvalue. For example:

```cpp
Eigen::Tensor<int, 1> a(3);
a.setValues({{100, 200, 300}});
Eigen::Tensor<int, 2> b(2, 3);
b.setZero();
b.chip(0, 0) = a;
std::cout << "a\n" << a << "\n";
std::cout << "b\n" << b << "\n";

// a
// 100
// 200
// 300

// b
//   100   200   300
//     0     0     0
```


The dimension can also be passed as a template parameter:

```cpp
b.chip<0>(1) = a;  // Equivalent to b.chip(1,0) = a;
```

Note that only one dimension can be chipped at a time.
To chip off multiple dimensions, you can chain calls

```cpp
Eigen::Tensor<int, 3> a(2, 3, 4);
Eigen::Tensor<int, 1> b = b.chip<2>(0) // Now has shape [2,3]
                           .chip<1>(0); // Now has shape [2]
```

Be careful in which order you chip, as each operation affects the shape of the intermediate result.
For example:

```cpp
// AVOID THIS
Eigen::Tensor<int, 1> c = b.chip<1>(0) // Now has shape [2,4]
                           .chip<1>(0); // Now has shape [2]
```

In general, it's more intuitive to chip from the outermost dimension first.


### (Operation) reverse(const ReverseDimensions& reverse)

Returns a view of the input tensor that reverses the order of the coefficients
along a subset of the dimensions.  The argument reverse is an array of boolean
values that indicates whether or not the order of the coefficients should be
reversed along each of the dimensions.  This operation preserves the dimensions
of the input tensor.

For example this is what happens when you `reverse()` the first dimension
of a 2D tensor:

```cpp
Eigen::Tensor<int, 2> a(4, 3);
a.setValues({{0, 100, 200}, {300, 400, 500},
            {600, 700, 800}, {900, 1000, 1100}});
Eigen::array<bool, 2> reverse{true, false};
Eigen::Tensor<int, 2> b = a.reverse(reverse);
std::cout << "a\n" << a << "\n";
std::cout << "b\n" << b << "\n";

// a
//    0   100   200
//  300   400   500
//  600   700   800
//  900  1000  1100
// b
//  900  1000  1100
//  600   700   800
//  300   400   500
//    0   100   200
```

### (Operation) roll(const Rolls& shifts)

Returns a tensor with the elements **circularly shifted** (like bit rotation) along one or more dimensions.

For each dimension `i`, the content is shifted by `shifts[i]` positions:

- A **positive shift** of `+s` moves each value to a **lower index** by `s`.
- A **negative shift** of `-s` moves each value to a **higher index** by `s`.

```cpp
Eigen::Tensor<int, 2> a(3, 4);
a.setValues({{ 1,  2,  3,  4},
             { 5,  6,  7,  8},
             { 9, 10, 11, 12}});

Eigen::array<Eigen::Index, 2> shifts = {1, -2};

Eigen::Tensor<int, 2> rolled = a.roll(shifts);

std::cout << "a\n" << a << "\n";
std::cout << "rolled\n" << rolled << "\n";

// a
// 1  2  3  4
// 5  6  7  8
// 9 10 11 12
//
// rolled
// 7  8  5  6
// 11 12  9 10
// 3  4  1  2
```

### (Operation) broadcast(const Broadcast& broadcast)

Returns a view of the input tensor in which the input is replicated one to many
times.
The broadcast argument specifies how many copies of the input tensor need to be
made in each of the dimensions.

```cpp
Eigen::Tensor<int, 2> a(2, 3);
a.setValues({{0, 100, 200}, {300, 400, 500}});
Eigen::array<int, 2> bcast{3, 2};
Eigen::Tensor<int, 2> b = a.broadcast(bcast);
std::cout << "a" << endl << a << endl << "b" << endl << b << endl;
// a
//    0   100   200
//  300   400   500
// b
//    0   100   200    0   100   200
//  300   400   500  300   400   500
//    0   100   200    0   100   200
//  300   400   500  300   400   500
//    0   100   200    0   100   200
//  300   400   500  300   400   500
```

Note: Broadcasting does not increase rank.
To broadcast into higher dimensions, you must first reshape the tensor with singleton (1) dimensions:

```cpp
Eigen::Tensor<int, 2> a(2, 3);
a.setValues({{0, 100, 200}, {300, 400, 500}});

Eigen::array<Eigen::Index, 3> new_shape = {1, 2, 3}; //Reshape to [1, 2, 3]
Eigen::array<int, 3> bcast = {4, 1, 1}; // Broadcast to [4, 2, 3]
Eigen::Tensor<int, 3> b = a.reshape(new_shape).broadcast(bcast);

std::cout << "b dimensions: " << b.dimensions() << "\n";
std::cout << b << "\n";
```

### (Operation) concatenate(const OtherDerived& other, Axis axis)

Returns a view of two tensors joined along a specified axis.
The dimensions of the two tensors must match on all axes except the concatenation axis.
The resulting tensor has the same rank as the inputs.

```cpp
Eigen::Tensor<int, 2> a(2, 3);
a.setValues({{0, 100, 200}, {300, 400, 500}});

Eigen::Tensor<int, 2> b(2, 3);
b.setValues({{-1, -2, -3}, {-4, -5, -6}});

// Concatenate along dimension 0: resulting shape is [4, 3]
Eigen::Tensor<int, 2> c = a.concatenate(b, 0);

// Concatenate along dimension 1: resulting shape is [2, 6]
Eigen::Tensor<int, 2> d = a.concatenate(b, 1);

std::cout << "a\n" << a << "\n"
          << "b\n" << b << "\n"
          << "c (concatenated along dim 0)\n" << c << "\n"
          << "d (concatenated along dim 1)\n" << d << "\n";
// a
//   0 100 200
// 300 400 500
// b
// -1 -2 -3
// -4 -5 -6
// c (concatenated along dim 0)
//   0 100 200
// 300 400 500
//  -1  -2  -3
//  -4  -5  -6
// d (concatenated along dim 1)
//   0 100 200  -1  -2  -3
// 300 400 500  -4  -5  -6
```

### (Operation)  pad(const PaddingDimensions& padding)

Returns a view of the input tensor in which the input is padded with zeros.

An optional second argument specifies the padding value (default is zero):
`a.pad(paddings, 42)` pads with the value 42.

```cpp
Eigen::Tensor<int, 2> a(2, 3);
a.setValues({{0, 100, 200}, {300, 400, 500}});
Eigen::array<pair<int, int>, 2> paddings;
paddings[0] = make_pair(0, 1);
paddings[1] = make_pair(2, 3);
Eigen::Tensor<int, 2> b = a.pad(paddings);
std::cout << "a" << endl << a << endl << "b" << endl << b << endl;
// a
//    0   100   200
//  300   400   500
// b
//    0     0     0    0
//    0     0     0    0
//    0   100   200    0
//  300   400   500    0
//    0     0     0    0
//    0     0     0    0
//    0     0     0    0
```

### (Operation) inflate(const Strides& strides)

Returns a tensor with zeros inserted between the elements of the input tensor
along each dimension. The `strides` array specifies the inflation factor for
each dimension: a stride of `s` inserts `s-1` zeros between consecutive
elements in that dimension. A stride of 1 leaves the dimension unchanged.

The output dimension sizes are `(input_dim - 1) * stride + 1`.

```cpp
Eigen::Tensor<float, 2> a(2, 3);
a.setValues({{1, 2, 3}, {4, 5, 6}});
Eigen::array<Eigen::Index, 2> strides{{2, 3}};
Eigen::Tensor<float, 2> b = a.inflate(strides);
std::cout << "b dimensions: " << b.dimension(0) << " x " << b.dimension(1) << "\n";
std::cout << "b\n" << b << "\n";

// b dimensions: 3 x 7
// b
// 1 0 0 2 0 0 3
// 0 0 0 0 0 0 0
// 4 0 0 5 0 0 6
```

This is the adjoint of the `stride()` operation and is useful for implementing
transposed convolutions (deconvolutions).

### (Operation)  extract_patches(const PatchDims& patch_dims)

Returns a tensor of coefficient patches extracted from the input tensor, where
each patch is of dimension specified by `patch_dims`. The returned tensor has
one greater dimension than the input tensor, which is used to index each patch.
The patch index in the output tensor depends on the data layout of the input
tensor: the patch index is the last dimension `ColMajor` layout, and the first
dimension in `RowMajor` layout.

For example, given the following input tensor:

```cpp
Eigen::Tensor<float, 2, DataLayout> tensor(3,4);
tensor.setValues({{0.0f, 1.0f, 2.0f, 3.0f},
                  {4.0f, 5.0f, 6.0f, 7.0f},
                  {8.0f, 9.0f, 10.0f, 11.0f}});

std::cout << "tensor: " << endl << tensor << endl;

// tensor:
//  0   1   2   3
//  4   5   6   7
//  8   9  10  11
```

Six 2x2 patches can be extracted and indexed using the following code:

```cpp
Eigen::Tensor<float, 3, DataLayout> patch;
Eigen::array<ptrdiff_t, 2> patch_dims;
patch_dims[0] = 2;
patch_dims[1] = 2;
patch = tensor.extract_patches(patch_dims);
for (int k = 0; k < 6; ++k) {
  std::cout << "patch index: " << k << endl;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      if (DataLayout == ColMajor) {
        std::cout << patch(i, j, k) << " ";
      } else {
        std::cout << patch(k, i, j) << " ";
      }
    }
    std::cout << endl;
  }
}
```

This code results in the following output when the data layout is `ColMajor`:

    patch index: 0
    0 1
    4 5
    patch index: 1
    4 5
    8 9
    patch index: 2
    1 2
    5 6
    patch index: 3
    5 6
    9 10
    patch index: 4
    2 3
    6 7
    patch index: 5
    6 7
    10 11

This code results in the following output when the data layout is RowMajor:

**NOTE**: the set of patches is the same as in `ColMajor`, but are indexed differently

    patch index: 0
    0 1
    4 5
    patch index: 1
    1 2
    5 6
    patch index: 2
    2 3
    6 7
    patch index: 3
    4 5
    8 9
    patch index: 4
    5 6
    9 10
    patch index: 5
    6 7
    10 11

### (Operation)  extract_image_patches(const Index patch_rows, const Index patch_cols, const Index row_stride, const Index col_stride, const PaddingType padding_type)

Returns a tensor of coefficient image patches extracted from the input tensor,
which is expected to have dimensions ordered as follows (depending on the data
layout of the input tensor, and the number of additional dimensions 'N'):

- `ColMajor`
    - 1st dimension: channels (of size d)
    - 2nd dimension: rows (of size r)
    - 3rd dimension: columns (of size c)
    - 4th-Nth dimension: time (for video) or batch (for bulk processing).

* `RowMajor` (reverse order of `ColMajor`)
    - 1st-Nth dimension: time (for video) or batch (for bulk processing).
    - N+1'th dimension: columns (of size c)
    - N+2'th dimension: rows (of size r)
    - N+3'th dimension: channels (of size d)

The returned tensor has one greater dimension than the input tensor, which is
used to index each patch. The patch index in the output tensor depends on the
data layout of the input tensor: the patch index is the 4'th dimension in
`ColMajor` layout, and the 4'th from the last dimension in `RowMajor` layout.

For example, given the following input tensor with the following dimension
sizes:
- depth:   2
- rows:    3
- columns: 5
- batch:   7

```cpp
Tensor<float, 4> tensor(2,3,5,7);
Tensor<float, 4, RowMajor> tensor_row_major = tensor.swap_layout();
```

2x2 image patches can be extracted and indexed using the following code:

#### 2D patch: `ColMajor` (patch indexed by second-to-last dimension)

```cpp
Tensor<float, 5> twod_patch;
twod_patch = tensor.extract_image_patches<2, 2>();
// twod_patch.dimension(0) == 2
// twod_patch.dimension(1) == 2
// twod_patch.dimension(2) == 2
// twod_patch.dimension(3) == 3*5
// twod_patch.dimension(4) == 7
```

#### 2D patch: `RowMajor` (patch indexed by the second dimension)

```cpp
Tensor<float, 5, RowMajor> twod_patch_row_major;
twod_patch_row_major = tensor_row_major.extract_image_patches<2, 2>();
// twod_patch_row_major.dimension(0) == 7
// twod_patch_row_major.dimension(1) == 3*5
// twod_patch_row_major.dimension(2) == 2
// twod_patch_row_major.dimension(3) == 2
// twod_patch_row_major.dimension(4) == 2
```

## Generation and Custom Operations

### (Operation) generate(const Generator& generator)

Returns a tensor whose values are computed by the given generator functor based
on element coordinates. The generator must define `operator()` taking an
`array<Index, NumDims>` of coordinates and returning a `Scalar`.

```cpp
// Generator that produces the linear index of each element.
template <typename Index, int NumDims>
struct LinearIndexGenerator {
  Eigen::array<Index, NumDims> dims_;
  LinearIndexGenerator(const Eigen::array<Index, NumDims>& dims) : dims_(dims) {}

  float operator()(const Eigen::array<Index, NumDims>& coords) const {
    float idx = 0;
    float stride = 1;
    for (int i = 0; i < NumDims; ++i) {
      idx += coords[i] * stride;
      stride *= dims_[i];
    }
    return idx;
  }
};

Eigen::Tensor<float, 2> t(3, 4);
Eigen::array<Eigen::Index, 2> dims{{3, 4}};
Eigen::Tensor<float, 2> result = t.generate(LinearIndexGenerator<Eigen::Index, 2>(dims));
```

### (Operation) customOp(const CustomUnaryFunc& func)

Applies a custom operation that can produce output with different dimensions
than the input. Unlike `unaryExpr()` which is element-wise, `customOp()`
gives full control over how the output is computed.

The functor must implement:
- `dimensions(const InputType& input)` — returns the output dimensions.
- `eval(const InputType& input, OutputType& output, const Device& device)` —
  computes the result.

```cpp
struct RowSumOp {
  // Output is a 1D tensor with size equal to the number of rows.
  template <typename Input>
  Eigen::DSizes<Eigen::Index, 1> dimensions(const Input& input) const {
    return Eigen::DSizes<Eigen::Index, 1>(input.dimension(0));
  }

  template <typename Input, typename Output, typename Device>
  void eval(const Input& input, Output& output, const Device& device) const {
    Eigen::array<Eigen::Index, 1> reduce_dims{{1}};
    output.device(device) = input.sum(reduce_dims);
  }
};

Eigen::Tensor<float, 2> a(3, 4);
a.setRandom();
Eigen::Tensor<float, 1> row_sums = a.customOp(RowSumOp());
```

A binary variant is also available:
```cpp
Eigen::Tensor<float, 2> result = a.customOp(b, MyBinaryCustomOp());
```

### (Operation) nullaryExpr(const CustomNullaryOp& func)

Creates a tensor from a custom nullary functor. The functor is called for
each element position.


## Special Operations

### (Operation) cast\<T\>()

Returns a tensor of type `T` with the same dimensions as the original tensor.
The returned tensor contains the values of the original tensor converted to
type `T`.

```cpp
Eigen::Tensor<float, 2> a(2, 3);
Eigen::Tensor<int, 2> b = a.cast<int>();
```

This can be useful for example if you need to do element-wise division of
Tensors of integers.
This is not currently supported by the Tensor library
but you can easily cast the tensors to floats to do the division:

```cpp
Eigen::Tensor<int, 2> a(2, 3);
a.setValues({{0, 1, 2}, {3, 4, 5}});
Eigen::Tensor<int, 2> b =
    (a.cast<float>() / a.constant(2).cast<float>()).cast<int>();
std::cout << "a\n" << a << "\n";
std::cout << "b\n" << b << "\n";

// a
// 0 1 2
// 3 4 5
//
// b
// 0 0 1
// 1 2 2
```

### (Operation)     eval()
See **Calling eval()**.



## Tensor Printing
Tensors can be printed into a stream object (e.g. `std::cout`) using different formatting options.

```cpp
Eigen::Tensor<float, 3> tensor3d(4, 3, 2);
tensor3d.setValues( {{{1, 2},
                      {3, 4},
                      {5, 6}},
                     {{7, 8},
                      {9, 10},
                      {11, 12}},
                     {{13, 14},
                      {15, 16},
                      {17, 18}},
                     {{19, 20},
                      {21, 22},
                      {23, 24}}} );
std::cout << tensor3d.format(Eigen::TensorIOFormat::Plain()) << "\n";
//  1  2
//  3  4
//  5  6
//
//  7  8
//  9 10
// 11 12
//
// 13 14
// 15 16
// 17 18
//
// 19 20
// 21 22
// 23 24
```

In the example, we used the predefined format `Eigen::TensorIOFormat::Plain`.
Here is the list of all predefined formats from which you can choose:
- `Eigen::TensorIOFormat::Plain()` for a plain output without braces. Different submatrices are separated by a blank line.
- `Eigen::TensorIOFormat::Numpy()` for numpy-like output.
- `Eigen::TensorIOFormat::Native()` for a `c++` like output which can be directly copy-pasted to `setValues()`.
- `Eigen::TensorIOFormat::Legacy()` for a backwards compatible printing of tensors.

If you send the tensor directly to the stream the default format is called which is `Eigen::TensorIOFormat::Plain()`.

You can define your own format by explicitly providing a `Eigen::TensorIOFormat` class instance. Here, you can specify:
- The overall prefix and suffix with `std::string tenPrefix` and `std::string tenSuffix`
- The prefix, separator and suffix for each new element, row, matrix, 3d subtensor, ... with `std::vector<std::string> prefix`, `std::vector<std::string> separator` and `std::vector<std::string> suffix`. Note that the first entry in each of the vectors refer to the last dimension of the tensor, e.g. `separator[0]` will be printed between adjacent elements,  `separator[1]` will be printed between adjacent matrices, ...
- `char fill`: character which will be placed if the elements are aligned.
- `int precision`
- `int flags`: an OR-ed combination of flags, the default value is 0, the only currently available flag is `Eigen::DontAlignCols` which allows to disable the alignment of columns, resulting in faster code.

## Interop with Eigen Matrix and Vector Types

Tensor data can be wrapped as an Eigen `Map<Matrix>`, and vice versa, Eigen
dense matrix/vector data can be wrapped as a `TensorMap`. This is a zero-copy
operation that simply reinterprets the underlying memory.

### Wrapping a Tensor as a Matrix

```cpp
Eigen::Tensor<float, 2> tensor(3, 4);
tensor.setRandom();

// View the tensor's data as an Eigen Matrix (no copy).
Eigen::Map<Eigen::MatrixXf> matrix(tensor.data(), 3, 4);
std::cout << "Matrix view:\n" << matrix << "\n";

// Modifications through the map are reflected in the tensor.
matrix(0, 0) = 42.0f;
assert(tensor(0, 0) == 42.0f);
```

### Wrapping a Matrix as a Tensor

```cpp
Eigen::MatrixXf matrix(3, 4);
matrix.setRandom();

// View the matrix's data as a rank-2 Tensor (no copy).
Eigen::TensorMap<Eigen::Tensor<float, 2>> tensor(matrix.data(), 3, 4);
std::cout << "Tensor view:\n" << tensor << "\n";

// You can also reshape to a different rank.
Eigen::TensorMap<Eigen::Tensor<float, 1>> flat(matrix.data(), 12);
```

**Important**: Both the Map and TensorMap are non-owning views. The underlying
data must remain valid for the lifetime of the view. Also note that the default
storage order of Eigen matrices is `ColMajor`, which matches the default Tensor
layout.

## Representation of scalar values

Scalar values are often represented by tensors of size 1 and rank 0.

For example `Tensor<T, N>::maximum()` returns a `Tensor<T, 0>`.

Similarly, the inner product of 2 1d tensors (through contractions) returns a 0d tensor.

The scalar value can be extracted as explained in **Reduction along all dimensions**.


## Limitations

*   The number of tensor dimensions is currently limited to 250.
*   On GPUs only floating point values are properly tested and optimized for.
*   C++14 or later is required to use the Tensor module.
