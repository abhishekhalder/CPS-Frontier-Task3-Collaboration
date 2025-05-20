include("../src/FrechetDistance.jl")

using .FrechetDistance
using Test
using Printf
using CSV
using Tables
using Distributions
using Debugger

PATHS_PATH = "../../../kbm_sim/sim_paths/"

@testset "Input validity" begin
    P = Float64[
        0 0
        0 1
    ] |> transpose

    Q = Float64[
        0 0 0
        0 0 1
    ] |> transpose

    @test_throws DimensionMismatch frechet(P, Q)
end

@testset "Basic Correctness" begin
    P = Float64[
        0 0 0
        0 0 1
        0 0 2
    ] |> transpose

    Q = Float64[
        0 0 0
        0 0 1
        0 1 1
        0 0 1
        0 0 2
    ] |> transpose

    @test frechet(P, P) == 0
    @test frechet(Q, Q) == 0
    @test frechet(P, Q) == 1
end

@testset "GP Paths Correctness" begin
	# Load three of our GP-generated paths
	P0 = CSV.File(PATHS_PATH*"path_wypts_0.txt") |> Tables.matrix |> transpose; P0=P0[1:2,:];
	P1 = CSV.File(PATHS_PATH*"path_wypts_1.txt") |> Tables.matrix |> transpose; P1=P1[1:2,:];
	P2 = CSV.File(PATHS_PATH*"path_wypts_2.txt") |> Tables.matrix |> transpose; P2=P2[1:2,:];

	# Gaussian noise parameters
	mu = 0;
	sigma = 0.5;
	min_thresh = -0.25;
	max_thresh = 0.25;

	# Add some Gaussian noise to paths
	# noise = rand(Truncated(Normal(mu, sigma), min_thresh, max_thresh), 1, 199) |> transpose
	# P0_n = copy(P0); P0_n[2,:] += noise;
	# P1_n = copy(P1); P1_n[2,:] += noise;
	# P2_n = copy(P2); P2_n[2,:] += noise;
	
	# Add some Gaussian noise to paths
	P0_n = copy(P0); P0_n[2,:] += rand(Truncated(Normal(mu, sigma), min_thresh, max_thresh), 1, 199) |> transpose;
	P1_n = copy(P1); P1_n[2,:] += rand(Truncated(Normal(mu, sigma), min_thresh, max_thresh), 1, 199) |> transpose;
	P2_n = copy(P2); P2_n[2,:] += rand(Truncated(Normal(mu, sigma), min_thresh, max_thresh), 1, 199) |> transpose;

	# Calculate Frechet distance between each path + noise and all original paths
	f00 = frechet(P0,P0_n)
	f10 = frechet(P1,P0_n)
	f20 = frechet(P2,P0_n)

	f01 = frechet(P0,P1_n)
	f11 = frechet(P1,P1_n)
	f21 = frechet(P2,P1_n)

	f02 = frechet(P0,P2_n)
	f12 = frechet(P1,P2_n)
	f22 = frechet(P2,P2_n)

	# Print results
	@printf("\n----------\n")
	@printf("frechet(P0,P0_n)=%.20f\n", f00)
	@printf("frechet(P1,P0_n)=%.20f\n", f10)
	@printf("frechet(P2,P0_n)=%.20f\n", f20)
	@printf("----------\n")
	@printf("frechet(P0,P1_n)=%.20f\n", f01)
	@printf("frechet(P1,P1_n)=%.20f\n", f11)
	@printf("frechet(P2,P1_n)=%.20f\n", f21)
	@printf("----------\n")
	@printf("frechet(P0,P2_n)=%.20f\n", f02)
	@printf("frechet(P1,P2_n)=%.20f\n", f12)
	@printf("frechet(P2,P2_n)=%.20f\n", f22)
	@printf("----------\n\n")

	# Verify that path with noise is closest to itself w/o noise
	@test f00 == min(f00, f10, f20)
	@test f11 == min(f01, f11, f21)
	@test f22 == min(f02, f12, f22)
end

