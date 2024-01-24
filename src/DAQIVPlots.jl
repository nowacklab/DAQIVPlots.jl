module DAQIVPlots

using Mmap
using StaticArrays
using JSON
using NPZ
using Roots
using GLMakie
using Unitful
using Statistics
using KernelDensity

struct OutCoefficients
    c::SVector{2, Float64}
end

buffer(c::OutCoefficients) = c.c

struct OutInverseCoefficients
    c::SVector{2, Float64}
end

buffer(c::OutInverseCoefficients) = c.c

inv(c::OutCoefficients) = OutInverseCoefficients([-c.c[1], one(eltype(c.c))] ./ c.c[2])
out_inv_coeff(c) = Ref(buffer(inv(OutCoefficients(c))))
out_voltage(c, sample) = c[1] + c[2] * sample

struct InCoefficients
    c::SVector{4, Float64}
end

buffer(c::InCoefficients) = c.c

in_coeff(c) = buffer(InCoefficients(c))
in_voltage(c, sample) = c[1] + c[2] * sample + c[3] * sample^2 + c[4] * sample^3

function main(dn = 0, plot = true, savefig = true)
    dirs = sort([dir for dir in readdir(sort = false) if startswith(dir, "daqiv-")], rev = true)
    recent = dirs[1 - dn]
    println(recent)
    contents = readdir(recent, sort = true)
    expected_contents = [
                         "input-samples.bin",
                         "output-samples.npy",
                         "parameters.json",
                        ]
    if !issubset(expected_contents, contents)
        error("Directory $(recent) does not contain the expected contents $(expected_contents)")
    end

    parameters = JSON.parsefile(joinpath(recent, "parameters.json"))

    println(parameters["comment"])

    bias_resistance = parameters["daqiv"]["daqTriangleCurrentFromZero"]["totalResistanceOhms"] * u"Ω"
    outinvcoeff = out_inv_coeff(parameters["daqiv"]["daqio"]["output"]["calibration"]["coefficients"])
    out_samples = npzread(joinpath(recent, "output-samples.npy"))
    out_voltages = out_voltage.(outinvcoeff, out_samples) .* u"V"
    out_currents = out_voltages ./ bias_resistance

    preamp_gain = parameters["preamp"]["gain"]
    incoeff = in_coeff(parameters["daqiv"]["daqio"]["input"]["calibration"]["coefficients"])
    in_stream = open(joinpath(recent, "input-samples.bin"), "r")
    in_samples = mmap(in_stream, Vector{Int16})

    sweeps = parameters["daqiv"]["daqio"]["output"]["signal"]["regenerations"]
    quarter_sweep = div(length(out_samples), 4)
    transition_threshold = 5.0u"mV"

    positive_offset = 0*quarter_sweep
    positive_indices = (1:quarter_sweep) .+ positive_offset
    positive_transitions = zeros(Int, quarter_sweep)
    positive_transition_currents = zeros(eltype(out_currents), sweeps)
    missing_positive_transitions = 0
    positive_threshold_input_voltage = preamp_gain * ustrip(u"V", transition_threshold)
    positive_threshold_sample = find_zero(s -> in_voltage(incoeff, s) - positive_threshold_input_voltage, 0)

    negative_offset = 2*quarter_sweep
    negative_indices = (1:quarter_sweep) .+ negative_offset
    negative_transitions = zeros(Int, quarter_sweep)
    negative_transition_currents = zeros(eltype(out_currents), sweeps)
    missing_negative_transitions = 0
    negative_threshold_input_voltage = preamp_gain * ustrip(u"V", -transition_threshold)
    negative_threshold_sample = find_zero(s -> in_voltage(incoeff, s) - negative_threshold_input_voltage, 0)

    for sweep in 1:sweeps
        positive_sweep_indices = positive_indices .+ (sweep - 1) * length(out_samples)
        positive_transition_index = findfirst(in_samples[positive_sweep_indices] .> positive_threshold_sample)
        if isnothing(positive_transition_index)
            positive_transition_currents[sweep] = out_currents[1] # Sentinel value
            missing_positive_transitions += 1
        else
            positive_transition_currents[sweep] = out_currents[positive_offset + positive_transition_index]
            positive_transitions[positive_transition_index] += 1
        end
        
        negative_sweep_indices = negative_indices .+ (sweep - 1) * length(out_samples)
        negative_transition_index = findfirst(in_samples[negative_sweep_indices] .< negative_threshold_sample)
        if isnothing(negative_transition_index)
            negative_transition_currents[sweep] = out_currents[1] # Sentinel value
            missing_negative_transitions += 1
        else
            negative_transition_currents[sweep] = out_currents[negative_offset + negative_transition_index]
            negative_transitions[negative_transition_index] += 1
        end
    end

    close(in_stream)

    if plot
        min_transition = findfirst(positive_transitions .> 0)
        max_transition = findlast(positive_transitions .> 0)
        span = max_transition - min_transition
        margin = div(span, 8)
        plot_indices = (min_transition - margin):(max_transition + margin)
	dI = mean(out_currents[positive_offset .+ plot_indices[2:end]] .- out_currents[positive_offset .+ plot_indices[1:end-1]])
        fig, ax, plt = lines(
	    ustrip.(u"μA", out_currents[plot_indices .+ positive_offset]),
	    positive_transitions[plot_indices] ./ (ustrip(u"μA", dI) * sweeps),
            )
        ax.xlabel = rich("I = V / R", subscript("b"), " (μA)")
	ax.ylabel = rich("Transition probability density (μA", superscript("-1"), ")")
        ax.title = "$(recent)"

        positive_kde = kde_lscv(ustrip.(u"μA", positive_transition_currents))
        lines!(ax, positive_kde.x, positive_kde.density)
        
        #=
        fig, ax, plt = lines(
            ustrip.(u"μA", repeat(out_currents, sweeps)),
            ustrip.(u"mV", in_voltages[1:sweeps*length(out_currents)])
            )
        ax.xlabel = rich("I = V / R", subscript("b"), " (μA)")
        ax.ylabel = "V (mV)"
        ax.title = "$(recent)"
        =#

        display(fig)
        if savefig
            save("../plots/$(recent).png", fig, px_per_unit = 4)
        end
        #return parameters, (fig, ax, plt)
    end
    return parameters, positive_transitions, positive_kde, dI
end

export main

end # module DAQIVPlots
