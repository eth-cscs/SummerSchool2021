#ifndef STATS_H
#define STATS_H

namespace stats
{

extern unsigned long long flops_diff, flops_bc, flops_blas1;
extern unsigned int iters_cg, iters_newton;
extern bool verbose_output;

} // namespace stats

#endif // STATS_H

