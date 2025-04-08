import os
import numpy as np
from rdcheck_pops import get_errors, read_rdcheckfile

"""
Consider using natural populations in combination with other convergence measures, 
such as the cumulative deviation of observables

For complex systems like pyrazine, natural populations in the range of 0.005-0.01
at 150fs indicate that full convergence of the autocorrelation function may not have been
achieved.

"""


def read_populations(dir, name, n_states, no_err_read=False, runrdcheck=True):

    if no_err_read:
        std_pops = read_rdcheckfile(name, dir, n_states, runrdcheck=runrdcheck)
        return std_pops

    errors, std_pops, eff_pops = get_errors(
        dir, system=name, n_states=n_states, return_pop_series=True, runrdcheck=runrdcheck
    )

    return std_pops, eff_pops, errors


def check_convergence(dir, name, spf_range, n_states, ctol=0.1, plotting=True):

    # compute RMSE in standard_pops and previous standard_pops
    # compute RMSE in effective_pops and previous effective_pops

    # if both RMSEs are below convergence threshold, return True
    # else, continue to next n_spf and check for convergence.

    pop_series_std = []
    pop_series_eff = []

    for idx, n_spf in enumerate(spf_range):

        if n_spf == "exact":
            errors, standard_pops, effective_pops = get_errors(
                dir, system=f"{name}_exact", n_states=n_states, return_pop_series=True
            )
        else:
            errors, standard_pops, effective_pops = get_errors(
                dir, system=f"{name}_spf{n_spf}", n_states=n_states, return_pop_series=True
            )

        print(f"\nn_spf = {n_spf}")
        conv_err_std = 0
        conv_err_eff = 0
        if idx > 0:
            curr_std = list(standard_pops.values())
            prev_std = list(pop_series_std[-1].values())
            curr_eff = list(effective_pops.values())
            prev_eff = list(pop_series_eff[-1].values())

            for ti in range(len(curr_std)):
                d_std = sum(
                    [(curr_std[ti][idx] - prev_std[ti][idx]) ** 2 for idx in range(n_states)]
                )
                d_eff = sum(
                    [(curr_eff[ti][idx] - prev_eff[ti][idx]) ** 2 for idx in range(n_states)]
                )
                conv_err_std += d_std
                conv_err_eff += d_eff
            conv_err_std = np.sqrt(conv_err_std)
            conv_err_eff = np.sqrt(conv_err_eff)

            print(
                f"cumulative improvement going from n_spf={idx-1} to n_spf={idx} in observables (std): {conv_err_std}"
            )
            print(
                f"cumulative improvement going from n_spf={idx-1} to n_spf={idx} in observables (eff): {conv_err_eff}\n"
            )

            if conv_err_std < ctol and conv_err_eff < ctol:
                print(
                    f"Population converged at std: {round(conv_err_std,4)}, eff: {round(conv_err_eff,4)}"
                )
                print(f"(Tolerance = {ctol}).")
                return True, (conv_err_std, conv_err_eff), (standard_pops, effective_pops)

        pop_series_std.append(standard_pops)
        pop_series_eff.append(effective_pops)

    print(
        f"Did not find convergence (ctol = {ctol}) by n_spf = {spf_range[-1]}\nstd rmse: {round(conv_err_std,4)} \neff rmse: {round(conv_err_eff,4)}"
    )

    return False, (conv_err_std, conv_err_eff), (standard_pops, effective_pops)


# Function to get highest available spf number for a given deltaT and m. Returns the RMSE's with respect to previous
# n_spf as a measure of convergence.


def get_run_pops(dir, name, n_states):

    if os.path.isdir(f"{dir}/{name}"):
        print(f"Found directory: {dir}/{name}")
        # Now do a check if the run has been performed.
        speed_path = os.path.join(f"{dir}/{name}", "speed")
        if os.path.isfile(speed_path):
            found_std_run = True
    if os.path.isdir(f"dir_err"):
        print(f"Found directory: {dir}/{name}_err")
        # Now do a check if the run has been performed
        speed_path_err = os.path.join(f"{dir}/{name}_err", "speed")
        if os.path.isfile(speed_path_err):
            found_eff_run = True

    if found_std_run and found_eff_run:
        print("")

    is_conv, rmses, pops = check_convergence(
        dir, name, spf_range, n_states, ctol=0.1, plotting=True
    )
    std_rmses, eff_rmses = rmses
    return pops, rmses


def get_highest_spf_run(dir, name, spf_range, n_states):

    parent_dir = f"{dir}/{name}"
    highest = 0
    for n_spf in spf_range:
        found_std_run = False
        found_eff_run = False
        if os.path.isdir(f"{parent_dir}_spf{n_spf}"):
            print(f"Found directory: {parent_dir}_spf{n_spf}")
            # Now do a check if the run has been performed.
            speed_path = os.path.join(f"{parent_dir}_spf{n_spf}", "speed")
            if os.path.isfile(speed_path):
                found_std_run = True
        if os.path.isdir(f"{parent_dir}_spf{n_spf}_err"):
            print(f"Found directory: {parent_dir}_spf{n_spf}_err")
            # Now do a check if the run has been performed
            speed_path_err = os.path.join(f"{parent_dir}_spf{n_spf}_err", "speed")
            if os.path.isfile(speed_path_err):
                found_eff_run = True

        if found_std_run and found_eff_run:
            highest = n_spf

    is_conv, rmses, pops = check_convergence(
        dir, name, spf_range, n_states, ctol=0.1, plotting=True
    )
    std_rmses, eff_rmses = rmses
    print(
        f"Is converged?: {is_conv}\n Standard RMSE from previous: {std_rmses}\n Effective RMSE from previous: {eff_rmses}"
    )

    return highest, pops, rmses


def read_series(dir, name, m_set, max_spf, n_states):
    """
    This function, given a directory and base run name, will read the outputs of the MCTDH runs with `m=2,3,..., max_m`.
    The highest possible spf run will be used for extracting results for the particular `m`.

    Output: A dictionary with `m` being the key, and value being the standard and effective population time series
    for this `m`.

    """

    results = {}
    for m in m_set:
        name_m = f"{name}_{m}"
        # obtain best result for this m
        n_spf_highest, pop_series, conv_errs = get_highest_spf_run(
            dir, name_m, range(1, max_spf + 1), n_states
        )
        print(
            f"Using the n_spf = {n_spf_highest} results for m = {m} (convergence errors = {conv_errs})."
        )
        results[m] = pop_series

    return results


def get_error_metrics(std_popseries, eff_popseries, n_states, t_max=None):

    errors = {}

    """
    Returns:
        L1 error
        L2 error 
        max error 
    """

    if t_max is None:
        t_max = list(std_popseries.keys())[-1]

    for t in std_popseries.keys():

        if t > t_max:
            break

        pop_errors = np.zeros(n_states)
        std_pops_at_t = np.array(std_popseries[t])
        eff_pops_at_t = np.array(eff_popseries[t])

        diffs = np.abs(std_pops_at_t - eff_pops_at_t)

        for idx, diff in enumerate(diffs):
            pop_errors[idx] = diff
        errors[t] = pop_errors

    errors_total = [
        np.sum([errors[t][i] for t in errors.keys() if t <= t_max]) for i in range(n_states)
    ]
    errors_mean = [
        np.average([errors[t][i] for t in errors.keys() if t <= t_max]) for i in range(n_states)
    ]
    errors_var = [
        np.var([errors[t][i] for t in errors.keys() if t <= t_max]) for i in range(n_states)
    ]
    errors_max = [max([errors[t][i] for t in errors.keys() if t <= t_max]) for i in range(n_states)]
    errors_at_finalt = list(errors.values())[-1]
    print("Error analysis:--------")
    print(f"total: {errors_total}")
    print(f"mean: {errors_mean} \n    variances: {errors_var}")
    print(f"max: {errors_max}")
    return errors_total, errors_mean, errors_var, errors_max, errors_at_finalt


if __name__ == "__main__":

    deltaT_series = [0.1]
    mode_series = [2, 3, 4]
    n = 2

    namer = "exact"
    for deltaT in deltaT_series:
        for m in mode_series:

            name = f"no4a_{n}s{m}m"
            dir = f"./no4a_{n}s{m}m/deltaT={deltaT}"
            max_spf = 1

            # results = read_series(dir, name, max_m, max_spf, n_states)

            import matplotlib.pyplot as plt

            spf_range = [8]
            is_conv, conv_errs, pops = check_convergence(
                dir, name, ["exact"], n, ctol=0.1, plotting=True
            )

            std_pops, eff_pops = pops

            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

            d1_color = "blue"
            d2_color = "red"
            d3_color = "orange"
            d4_color = "green"

            d_colors = [d1_color, d2_color, d3_color, d4_color]

            d1_style = "solid"
            d2_style = "dashed"
            d3_style = "dashdot"
            d4_style = "dotted"

            d_styles = [d1_style, d2_style, d3_style, d4_style]

            errors = {}
            for t in std_pops.keys():
                errors[t] = np.abs(np.array(std_pops[t]) - np.array(eff_pops[t]))

            for idx in range(n):
                axes[0].plot(
                    list(std_pops.keys()),
                    [e[idx] for e in std_pops.values()],
                    label=f"d{idx+1}",
                    color=d_colors[idx],
                    linestyle=d_styles[idx],
                )
                axes[1].plot(
                    list(eff_pops.keys()),
                    [e[idx] for e in eff_pops.values()],
                    label=f"d{idx+1}",
                    color=d_colors[idx],
                    linestyle=d_styles[idx],
                )
                axes[2].plot(
                    list(errors.keys()),
                    [e[idx] for e in errors.values()],
                    label=f"d{idx+1}",
                    color=d_colors[idx],
                    linestyle=d_styles[idx],
                )

            # axes[0].plot(list(std_pops.keys()), [e[1] for e in std_pops.values()], label='d2', color=d2_color, linestyle=d1_style)
            # axes[0].plot(list(std_pops.keys()), [e[2] for e in std_pops.values()], label='d3', color=d3_color, linestyle=d1_style)
            # axes[0].plot(list(std_pops.keys()), [e[3] for e in std_pops.values()], label='d4', color=d4_color, linestyle=d1_style)

            # axes[1].plot(list(eff_pops.keys()), [e[0] for e in eff_pops.values()], label='d1', color=d1_color, linestyle=d1_style)
            # axes[1].plot(list(eff_pops.keys()), [e[1] for e in eff_pops.values()], label='d2', color=d2_color, linestyle=d1_style)
            # axes[1].plot(list(eff_pops.keys()), [e[2] for e in eff_pops.values()], label='d3', color=d3_color, linestyle=d1_style)
            # axes[1].plot(list(eff_pops.keys()), [e[3] for e in eff_pops.values()], label='d4', color=d4_color, linestyle=d1_style)

            axes[0].set_title("Standard population dynamics")
            axes[0].set_xlabel("Time (fs)")
            axes[0].set_ylabel("Population")
            axes[0].legend()

            axes[1].set_title(f"Effective population dynamics, deltaT={deltaT}")
            axes[1].set_xlabel("Time (fs)")
            axes[1].set_ylabel("Population")

            np.set_printoptions(suppress=True)

            # axes[2].plot(list(errors.keys()), [e[0] for e in errors.values()], label='d1', color=d1_color, linestyle=d1_style)
            # axes[2].plot(list(errors.keys()), [e[1] for e in errors.values()], label='d2', color=d2_color, linestyle=d1_style)
            # axes[2].plot(list(errors.keys()), [e[2] for e in errors.values()], label='d3', color=d3_color, linestyle=d1_style)
            # axes[2].plot(list(errors.keys()), [e[3] for e in errors.values()], label='d4', color=d4_color, linestyle=d1_style)
            axes[2].set_title("Error between standard and effective dynamics")

            errors_100fs = get_error_metrics(std_pops, eff_pops, n, t_max=100)
            errors_1000fs = get_error_metrics(std_pops, eff_pops, n, t_max=None)

            errors_100fs = [np.round(errs, 3) for errs in errors_100fs]
            errors_1000fs = [np.round(errs, 3) for errs in errors_1000fs]

            (
                errors_total_100fs,
                errors_mean_100fs,
                errors_var_100fs,
                errors_max_100fs,
                errors_at_finalt_100fs,
            ) = errors_100fs
            (
                errors_total_1000fs,
                errors_mean_1000fs,
                errors_var_1000fs,
                errors_max_1000fs,
                errors_at_finalt_1000fs,
            ) = errors_1000fs

            error_textbox = f"Error (1000fs): {errors_at_finalt_1000fs}\nTotal error (1000fs): {errors_total_1000fs} \nMax error (1000fs): {errors_max_1000fs} \nError (100fs): {errors_at_finalt_100fs}\nTotal error (100fs): {errors_total_100fs} \nMax error (100fs): {errors_max_100fs}"
            # axes[2].text(0, 1, error_textbox,
            #         transform=axes[2].transAxes,
            #         verticalalignment='top',
            #         horizontalalignment='left')

            # Set axis limits to ensure text is visible

            # Add text below the plot using annotate
            fig.subplots_adjust(bottom=0.3)
            fig.text(0.5, 0, error_textbox, ha="center", fontsize=10)

            plt.savefig(f"no4a_{n}s{m}m_mctdh_deltaT={deltaT}_{namer}.png")
    #         #plt.show()

    # series_err_tot = []
    # series_err_max = []
    # series_err_final = []

    # # for m in results:
    # #     print(f'********\nM = {m}')
    # #     errs_tot, errs_mean, errs_var, errs_max, errs_finalt = get_error_metrics(results[m][0], results[m][1], n_states)

    # #     series_err_tot.append(errs_tot)
    # #     series_err_max.append(errs_max)
    # #     series_err_final.append(errs_finalt)

    # #     #print(m)

    # # # n_states = 4
    # # # m = 2
    # # # t = 100
    # # print('Series of total error with m:')
    # # for err in series_err_tot:
    # #     print(max(err))

    # # print('Series of max error with m:')
    # # for err in series_err_max:
    # #     print(max(err))

    # # print('Errors at final t with m:')
    # # for err in series_err_final:
    # #     print(max(err))

    # # deltaT= 0.1
    # # m = 2
    # # n = 2
    # # name = f'no4a_{n}s{m}m_spf1'
    # # dir = f'./no4a_{n}s{m}m/deltaT={deltaT}'

    # # # n_states = 4
    # # # m=5
    # # # deltaT = 0.25
    # # # dir = f'no4a_deltaT={deltaT}'
    # # # name = f'no4a_{m}'
    # # spf_range = list(range(1,9))
    # # spf_range = [1]
    # # n_spf_max, pops, rmses = get_highest_spf_run(dir, name, spf_range, n_states=n_states)
    # # #is_conv, conv_errs, pops = check_convergence(dir, name, range(1, n_spf_max+1), n_states=n_states, ctol=0.1)

    # # print(pops)

    # # std_pops, eff_pops = pops

    # # #print(f'MCTDH converged? : {is_conv}')

    # # import matplotlib.pyplot as plt

    # # c0 = 'red'
    # # c1 = 'blue'
    # # c2 = 'green'
    # # c3 = 'purple'
    # # c = [c0,c1,c2,c3]

    # # for i in range(n_states):
    # #     plt.plot(list(std_pops.keys()), [p[i] for p in list(std_pops.values())], color=c[i],linestyle='-')
    # #     plt.plot(list(eff_pops.keys()), [p[i] for p in list(eff_pops.values())], color=c[i], linestyle=':')
    # # plt.show()
