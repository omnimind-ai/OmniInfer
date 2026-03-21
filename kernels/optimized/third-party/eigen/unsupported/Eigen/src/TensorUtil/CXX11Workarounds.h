// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2013 Christian Seiler <christian@iwakd.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11WORKAROUNDS_H
#define EIGEN_CXX11WORKAROUNDS_H

namespace Eigen {

namespace internal {

/* array_get overloads for std::vector, used by tensor code.
 */

template <std::size_t I_, class T>
constexpr T& array_get(std::vector<T>& a) {
  return a[I_];
}
template <std::size_t I_, class T>
constexpr T&& array_get(std::vector<T>&& a) {
  return a[I_];
}
template <std::size_t I_, class T>
constexpr T const& array_get(std::vector<T> const& a) {
  return a[I_];
}

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_CXX11WORKAROUNDS_H

/*
 * kate: space-indent on; indent-width 2; mixedindent off; indent-mode cstyle;
 */
