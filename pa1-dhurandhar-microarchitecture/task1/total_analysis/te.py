rows = [(512, 14.48, "simd256_v3", 17.39), (1024, 14.52, "simd256_v3", 18.10),
        (2048, 10.98, "simd256_v3", 15.33), (4096, 9.13, "simd256_v3", 14.18),
        (8192, 8.14, "simd128_v4", 14.01), (16384, 7.73, "simd128_v4", 8.11)]
with open("out/tab_combined_synergy.tex", "w") as f:
    f.write("\\begin{table}[htbp]\n\\centering\\small\n"
            "\\caption{The combined kernel against the best single technique "
            "at each size.}\n\\label{tab:combined-synergy}\n"
            "\\begin{tabular}{lrlrr}\n\\toprule\n"
            "Size & Best single & (variant) & Combined & Gain \\\\\n\\midrule\n")
    for n, best, var, opt in rows:
        f.write(f"{n}$\\times${n} & {best:.2f}$\\times$ & \\texttt{{{var}}} & "
                f"{opt:.2f}$\\times$ & {opt/best:.2f}$\\times$ \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")