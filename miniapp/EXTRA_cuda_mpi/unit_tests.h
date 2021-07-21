#pragma once

#include <cmath> // for std::fabs
#include <iostream> // for std::cout

#include "data.h"
#include "operators.h"

using namespace data;
using namespace linalg;
using namespace operators;

// Print a success/fail message for the output of a test.
bool test_output(bool success, const char* name) {
    if (domain.rank==0) {
        printf("%-25s : %-10s", name, (success? "\033[1;32mpassed\033[0m\n": "\033[1;31mfailed\033[0m\n"));
    }
    return success;
}

// Test whether expected is within epsilon of value, where epsilon is
// rel_tol*expected.
template <typename T>
bool check_value(T value, T expected, T rel_tol=1e-14) {
    T epsilon = std::max(rel_tol, rel_tol*std::fabs(expected));
    return std::fabs(value-expected)<epsilon;
}

// Input:
//  local success result (true or false)
// Output:
//  true if all ranks had true success
//  false if any ranks had false success
static
bool global_reduce(bool success) {
    int local_result = success? 1: 0;
    int global_result = 1;
    MPI_Allreduce(&local_result, &global_result, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    return global_result;
}

// Simple test for boundary exchange.
// Sets the value of the local field to equal the MPI rank, then performs and exchange.
// On completion, we expect that all values received from each neighbor match the neighbor's
// MPI rank.
static
bool test_exchange_rank() {
    const int rank = domain.rank;

    // set all local values to equal the rank
    ss_fill(x_old, rank);

    // set all boundary fields to -1
    ss_fill(bndN, -1);
    ss_fill(bndE, -1);
    ss_fill(bndS, -1);
    ss_fill(bndW, -1);

    // call exchange
    exchange_rdma(x_old);

    // copy gpu buffers to the host
    bndN.update_host();
    bndE.update_host();
    bndS.update_host();
    bndW.update_host();

    auto test = [&](int direction, data::Field const& f, const char* name) {
        bool success = true;
        double expected = direction;
        for (int i=0; i<f.length(); ++i) {
            success = check_value(f[i], expected);
            if (!success) {
                    std::cout << "  fail rank " << rank << ": location " << i << " on " << name << " boundary"
                              << ": expected " << expected << ", got " << f[i]
                              << std::endl;
                break;
            }
        }

        return global_reduce(success);
    };

    if (!test(domain.neighbour_north, bndN, "north")) return false;
    if (!test(domain.neighbour_east,  bndE, "east"))  return false;
    if (!test(domain.neighbour_south, bndS, "south")) return false;
    if (!test(domain.neighbour_west,  bndW, "west"))  return false;

    return true;
}

// Test for boundary exchange where the received input values vary spatially.
// buffers exchanged on the north-south vary in x dimension, and buffers
// exchange on the east-west vary in the y dimension.
static
bool test_exchange_spatial() {
    const int rank = domain.rank;
    const int nx = domain.nx;
    const int ny = domain.ny;

    // set all boundary fields to -1
    ss_fill(bndN, -1);
    ss_fill(bndS, -1);
    ss_fill(bndE, -1);
    ss_fill(bndW, -1);

    auto test = [&](int direction, data::Field const& f, const char* name) {
        bool success = true;
        if (direction>=0) {
            for (int i=0; i<f.length(); ++i) {
                success = check_value(f[i], double(i));
                if (!success) {
                    std::cout << "  fail rank " << rank << ": location " << i << " on " << name << " boundary"
                              << ": expected " << double(i) << ", got " << f[i]
                              << std::endl;
                    break;
                }
            }
        }

        return global_reduce(success);
    };

    // set input field values
    for (int i=0; i<nx; ++i) {
        for (int j=0; j<ny; ++j) {
            x_old(i,j) = i;
        }
    }
    x_old.update_device();

    // call exchange
    exchange_rdma(x_old);

    // copy gpu buffers to the host
    bndN.update_host();
    bndS.update_host();

    if (!test(domain.neighbour_north,  bndN, "north"))  return false;
    if (!test(domain.neighbour_south,  bndS, "south"))  return false;

    // set input field values
    for (int i=0; i<nx; ++i) {
        for (int j=0; j<ny; ++j) {
            x_old(i,j) = j;
        }
    }
    x_old.update_device();

    // call exchange
    exchange_rdma(x_old);

    // copy gpu buffers to the host
    bndE.update_host();
    bndW.update_host();

    if (!test(domain.neighbour_east,  bndE, "east"))  return false;
    if (!test(domain.neighbour_west,  bndW, "west"))  return false;

    return true;
}

// Test the distributed dot product.
static
bool test_dot() {
    const int n = xold.length();
    double expected = 2 * options.nx * options.ny;

    ss_fill(x_old, 1.);
    ss_fill(x_new, 2.);

    double result = ss_dot(x_old, x_new);

    bool success = check_value(result, expected);
    if (!success) {
        std::cout << "r" << domain.rank << ": dot expected " << expected
                  << ", got " << result << "\n";
    }
    return global_reduce(success);
}

// Test the distributed norm2.
static
bool test_norm2() {
    const int n = options.nx * options.ny;
    double expected = std::sqrt(double(2*2*n));

    ss_fill(x_old, 2.);

    double result = ss_norm2(x_old);

    bool success = check_value(result, expected);
    if (!success) {
        std::cout << "r" << domain.rank << ": dot expected " << expected
                  << ", got " << result << "\n";
    }
    return global_reduce(success);
}

// Run the unit tests.
// Returns false if any test fails.
// Stops after the first failing test.
static
bool unit_tests() {
    return
        test_output(test_exchange_rank(), "exchange rank") &&
        test_output(test_exchange_spatial(), "exchange spatial") &&
        test_output(test_dot(), "dot product") &&
        test_output(test_norm2(), "norm2");
}
