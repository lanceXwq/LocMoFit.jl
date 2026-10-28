using GLMakie

function generate_video(;
    data, model, unitmodel, states, losses, filename, framerate=8, ground_truth=nothing
)
    epoch = Observable(1)
    positions_observable = @lift begin
        X = LocMoFit.model_points(model, unitmodel, states[$epoch])
        [Point3f(x) for x in eachcol(X)]
    end

    loss_points = Observable(Point2f[])

    fig = Figure(; size=(1200, 400))
    ax1 = Axis3(fig[1, 1]; aspect=:data, xgridvisible=false, ygridvisible=false)
    ax2 = Axis3(
        fig[1, 2];
        aspect=:data,
        xgridvisible=false,
        ygridvisible=false,
        azimuth=0,
        elevation=0,
    )
    ax3 = Axis(
        fig[1, 3];
        title="Loss over iterations (BFGS)",
        xlabel="Epoch",
        ylabel="Loss",
        yscale=log10,
    )

    s3 = scatter!(ax1, data[1, :], data[2, :], data[3, :]; color=:gray, markersize=4)
    s1 = if !isnothing(ground_truth)
        scatter!(
            ax1,
            ground_truth[:, :x_nm],
            ground_truth[:, :y_nm],
            ground_truth[:, :z_nm];
            alpha=0.2,
        )
    end
    s2 = scatter!(ax1, positions_observable)
    # lims = extrema(data[1:3, :], dims=2)
    # limits!(ax1, 1.2 .* lims[1]..., 1.2 .* lims[2]..., 1.2 .* lims[3]...)
    if !isnothing(ground_truth)
        Legend(
            fig[0, 1],
            [s1, s2, s3],
            ["Ground truth", "Model points", "Localizations"];
            orientation=:horizontal,
        )
    else
        Legend(
            fig[0, 1], [s2, s3], ["Model points", "Localizations"]; orientation=:horizontal
        )
    end

    scatter!(ax2, data[1, :], data[2, :], data[3, :]; color=:gray, markersize=4)
    if !isnothing(ground_truth)
        scatter!(
            ax2,
            ground_truth[:, :x_nm],
            ground_truth[:, :y_nm],
            ground_truth[:, :z_nm];
            alpha=0.2,
        )
    end
    scatter!(ax2, positions_observable)
    # wireframe!(ax2, xs, ys, zs)
    # limits!(ax2, 1.2 .* lims[1]..., 1.2 .* lims[2]..., 1.2 .* lims[3]...)

    scatterlines!(ax3, loss_points; markersize=4)
    limits!(ax3, 0, length(losses), minimum(losses), maximum(losses) * 1.1)

    # Label(fig[0, :], text="Adam", fontsize=15)
    rowsize!(fig.layout, 0, Relative(0.05))
    record(fig, filename, 1:length(losses); framerate=framerate) do i
        epoch[] = i
        return loss_points[] = push!(loss_points[], Point2f(i, losses[i]))
    end
end
