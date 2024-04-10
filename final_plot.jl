using CairoMakie
using iPEPS
using JLD2
f2 = load("rspectral.jld2")
f1 = load("spin_spectral.jld2")

x1, y1 = plot_spectral(f1["x"], f1["y"]; factor = 0.1)
# x3, y3 = plot_spectral(f1["x"], f1["y"]; factor = 0.5)

# x2, y2 = plot_spectral(f2["x"], f2["y"]; factor = 0.5)
# x4, y4 = plot_spectral(f2["x"], f2["y"]; factor = 0.2)

f = Figure(fontsize = 35, xlabelfont = 34, ylabelfont = 34)
rowsize!(f.layout, 1, Aspect(1, 0.7))
ax1 = Axis(f[1, 1])# xlabel = L"{\omega} ~(J)", ylabel = "Intensity (a.u.)", xlabelsize=30, ylabelsize=30)
# ax2 = Axis(f[1, 1] )
lines!(ax1, x1, y1, color = (:black, 1), linewidth = 2)

# lines!(ax2, x4, y4, color = (:black, 1), linewidth = 2)

# lines!(ax2, x2, y2, color = (:red, 1) , linestyle = :dash, linewidth = 2.5)

# lines!(ax1, x3, y3, color = (:blue), linestyle = :dash, linewidth = 2.5)
# scatter!(ax, es[1:end], swk0[1:end], label = "Syy")
# axislegend(ax; labelsize=30)

# ax2.yaxisposition = :right
# ax2.yticklabelalign = (:left, :center)
# ax2.xticklabelsvisible = false
# ax2.xticklabelsvisible = false
# ax2.xlabelvisible = false
ax1.xgridvisible = false
ax1.ygridvisible = false
# ax2.xgridvisible = false
# ax2.ygridvisible = false

xlims!(ax1, low = 0, high = 6.5)
ylims!(ax1, low = 0, high = 30)
# xlims!(ax2, low = 0, high = 12.3)
# ylims!(ax2, low = 0, high = 8.5)
ax1.xtickalign = 1
# ax2.xtickalign = 1
ax1.ytickalign = 1
# ax2.ytickalign = 1
ax1.xticks = 0:2:12
# ax2.xticks = 0:2:12
# ax1.yticklabelcolor = :blue
# ax2.yticklabelcolor = :red

ax1.yticks = 0:8:30
# ax2.yticks = 0:1:4
resize_to_layout!(f)
f
save("spin_q0.pdf", f, pt_per_unit = 1)