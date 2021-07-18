#include "linalg.h"

using data::Field;

template <typename F>
bool run_test(F f, const char* name) {
    auto success = f();
    printf("%-25s : ", name);
    if(!success) {
        printf("\033[1;31mfailed\033[0m\n");
        return false;
    }
    printf("\033[1;32mpassed\033[0m\n");
    return true;
} 
template <typename T>
bool check_value(T value, T expected, T tol) {
    if(std::fabs(value-expected)>tol) {
        std::cout << "  expected " << expected << " got " << value << std::endl;
        return false;
    }
    return true;
}

bool test_scaled_diff() {
    auto n = 5;
    Field y(n,1);
    Field l(n,1);
    Field r(n,1);

    for(auto i=0; i<n; ++i) {
        l[i] = 7.0;
        r[i] = 2.0;
    }
    l.update_device();
    r.update_device();

    linalg::ss_scaled_diff(y, 2.0, l, r);
    y.update_host();

    bool status = true;
    for(auto i=0; i<n; ++i) {
        status = status && check_value(y[i], 10.0, 1.e-13);
    }
    return status;
}

bool test_fill() {
    auto n = 5;
    Field x(n,1);

    for(auto i=0; i<n; ++i) {
        x[i] = 3.0;
    }
    x.update_device();

    linalg::ss_fill(x, 2.0);
    x.update_host();

    bool status = true;
    for(auto i=0; i<n; ++i) {
        status = status && check_value(x[i], 2.0, 1.e-13);
    }
    return status;
}

bool test_axpy() {
    auto n = 5;
    Field x(n,1);
    Field y(n,1);

    for(auto i=0; i<n; ++i) {
        x[i] = 3.0;
        y[i] = 5.0;
    }
    x.update_device();
    y.update_device();

    linalg::ss_axpy(y, 0.5, x);
    y.update_host();

    bool status = true;
    for(auto i=0; i<n; ++i) {
        status = status && check_value(y[i], (0.5*3.0 + 5.0), 1.e-13);
    }
    return status;
}

bool test_add_scaled_diff() {
    auto n = 5;
    Field y(n,1);
    Field x(n,1);
    Field l(n,1);
    Field r(n,1);

    for(auto i=0; i<n; ++i) {
        x[i] = 3.0;
        l[i] = 7.0;
        r[i] = 2.0;
    }
    x.update_device();
    l.update_device();
    r.update_device();

    linalg::ss_add_scaled_diff(y, x, 1.5, l, r);
    y.update_host();

    bool status = true;
    for(auto i=0; i<n; ++i) {
        status = status && check_value(y[i], 3. + 1.5 * (7. - 2.), 1.e-13);
    }
    return status;
}

bool test_scale() {
    auto n = 5;
    Field x(n,1);
    Field y(n,1);

    for(auto i=0; i<n; ++i) {
        x[i] = 3.0;
    }
    x.update_device();

    linalg::ss_scale(y, 0.5, x);
    y.update_host();

    bool status = true;
    for(auto i=0; i<n; ++i) {
        status = status && check_value(y[i], 1.5, 1.e-13);
    }
    return status;
}

bool test_lcomb() {
    auto n = 5;
    Field x(n,1);
    Field y(n,1);
    Field z(n,1);

    for(auto i=0; i<n; ++i) {
        x[i] = 3.0;
        z[i] = 7.0;
    }
    x.update_device();
    z.update_device();

    linalg::ss_lcomb(y, 0.5, x, 2.0, z);
    y.update_host();

    bool status = true;
    for(auto i=0; i<n; ++i) {
        status = status && check_value(y[i], (0.5*3. + 2.*7.), 1.e-13);
    }
    return status;
}

bool test_copy() {
    auto n = 5;
    Field x(n,1);
    Field y(n,1);

    for(auto i=0; i<n; ++i) {
        x[i] = 3.0;
    }

    x.update_device();
    linalg::ss_copy(y, x);
    y.update_host();

    bool status = true;
    for(auto i=0; i<n; ++i) {
        status = status && check_value(y[i], x[i], 1.e-13);
    }

    return status;
}

bool test_dot() {
    auto n = 5;
    Field x(n,1);
    Field y(n,1);

    for(auto i=0; i<n; ++i) {
        x[i] = 3.0;
        y[i] = 7.0;
    }
    x.update_device();
    y.update_device();

    auto result = linalg::ss_dot(x, y);

    return check_value(result, n*3.*7., 1.e-13);
}

bool test_norm2() {
    auto n = 5;
    Field x(n,1);

    for(auto i=0; i<n; ++i) {
        x[i] = 2.0;
    }
    x.update_device();

    auto result = linalg::ss_norm2(x);

    return check_value(result, sqrt(2.0 * 2.0 * 5.0), 1.e-13);
}

////////////////////////////////////////////////////////////////////////////////
// main
////////////////////////////////////////////////////////////////////////////////
int main(void) {
    run_test(test_dot,          "ss_dot");
    run_test(test_norm2,        "ss_norm2");
    run_test(test_scaled_diff,  "ss_scaled_diff");
    run_test(test_fill,         "ss_fill");
    run_test(test_axpy,         "ss_axpy");
    run_test(test_add_scaled_diff, "ss_add_scaled_diff");
    run_test(test_scale,        "ss_scale");
    run_test(test_lcomb,        "ss_lcomb");
    run_test(test_copy,         "ss_copy");
}

