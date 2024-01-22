module DAQIVPlots

using StaticArrays
using JSON
using NPZ
using GLMakie
using Unitful
using Statistics

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
out_voltage(c, sample::Int16) = c[1] + c[2] * sample

struct InCoefficients
    c::SVector{4, Float64}
end

buffer(c::InCoefficients) = c.c

in_coeff(c) = Ref(buffer(InCoefficients(c)))
in_voltage(c, sample::Int16) = c[1] + c[2] * sample + c[3] * sample^2 + c[4] * sample^3

function main(dn = 0)
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

    bias_resistance = parameters["daqiv"]["daqTriangleCurrentFromZero"]["totalResistanceOhms"] * u"Ω"
    outinvcoeff = out_inv_coeff(parameters["daqiv"]["daqio"]["output"]["calibration"]["coefficients"])
    out_samples = npzread(joinpath(recent, "output-samples.npy"))
    out_voltages = out_voltage.(outinvcoeff, out_samples) .* u"V"
    out_currents = out_voltages ./ bias_resistance

    preamp_gain = parameters["preamp"]["gain"]
    incoeff = in_coeff(parameters["daqiv"]["daqio"]["input"]["calibration"]["coefficients"])
    in_samples = reinterpret(Int16, read(joinpath(recent, "input-samples.bin")))
    in_voltages = in_voltage.(incoeff, in_samples) ./ preamp_gain .* u"V"

    resistance = mean(abs.(in_voltages[1:length(out_currents)] ./ out_currents)) |> u"Ω"
    println("Resistance: $(resistance)")

    #sweeps = parameters["daqiv"]["daqio"]["output"]["signal"]["regenerations"]
    sweeps = 1
    fig, ax, plt = lines(
        ustrip.(u"μA", repeat(out_currents, sweeps)),
        ustrip.(u"μV", in_voltages[1:sweeps*length(out_currents)])
        )
    ax.xlabel = rich("I = V / R", subscript("b"), " (μA)")
    ax.ylabel = "V (μV)"

    display(fig)
    return parameters
end

export main

end # module DAQIVPlots
