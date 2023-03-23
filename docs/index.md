# Introduction

## Overview

This document will serve as an introduction to and usage example for the
[libplacebo](https://code.videolan.org/videolan/libplacebo) API. This is not
intended as a full API reference, for that you should see the repository of
[header
files](https://code.videolan.org/videolan/libplacebo/-/tree/master/src/include/libplacebo),
which are written to be (hopefully) understandable as-is.

libplacebo exposes large parts of its internal abstractions publicly. This
guide will take the general approach of starting as high level as possible and
diving into the details in later chapters.

A full listing of currently available APIs and their corresponding header
files can be seen
[here](https://code.videolan.org/videolan/libplacebo#api-overview).

## Getting Started

To get started using libplacebo, you need to install it (and its development
headers) somehow onto your system. On most distributions, this should be as
simple as installing the corresponding `libplacebo-devel` package, or the
appropriate variants.

You can see a fill list of libplacebo packages and their names [on
repology](https://repology.org/project/libplacebo/versions).

!!! note "API versions"

    This document is targeting the "v4 API" overhaul, and as such, examples
    provided will generally fail to compile on libplacebo versions below v4.x.

Alternatively, you can install it from the source code. For that, see the
build instructions [located here](https://code.videolan.org/videolan/libplacebo#installing).
