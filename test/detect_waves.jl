using FrequencyAdjustment
using Test
using FFTW
using LinearAlgebra

@testset "Wave Detection" begin
    function radial_chirp(; dims = [256, 256], periods = 50)
        N = length(dims)
        L = sqrt(periods / N)
        phase = 0
        for i=1:N
            v = ones(Int, N)
            v[i] = dims[i]
            phase = phase .+ (range(-L, L; length=dims[i]) |> z -> reshape(z, Tuple(v))).^2
        end
        cos.( 2π .* phase ) * 0.5 .+ 0.5
    end
    
    # Test FrequencyAdjustment.quadratic_lsqfit_log_abs
    @testset "Fit" begin
        F = [ -0.15239871856975162334+0.21952205672543895210im 0.08919925231415833100-0.16766123151155784554im -0.01182833868319268372+0.11445694576792818375im;
               0.15682785235079998998-0.20403648764122422898im -0.10762162541650523162+0.23128017716691728900im 0.01062975527894749472-0.18891857879292531242im;
              -0.14129975752905388209+0.13701704967141220548im 0.11663072407938072927-0.20707743005690823490im -0.02148254897458338336+0.18283270965969278254im]

        expected_result = (
            [-0.14890248386698482 0.09696823193834164; 0.09696823193834164 -0.18684814456493398], 
            [0.0049007478942518405, 0.4242494187584156], 
            -1.6545500501419337
        )

        result = FrequencyAdjustment.quadratic_lsqfit_log_abs(F)

        @test all(isapprox.(expected_result, result, atol=1e-10))
    end
    
    # Test FrequencyAdjustment.quadratic_critical_point
    @testset "Quadratic critical point location" begin
        Q, L, C = (
            Symmetric([-0.14890248386698482 0.09696823193834164; 0.09696823193834164 -0.18684814456493398]), 
            [0.0049007478942518405, 0.4242494187584156], 
            -1.6545500501419337
        )

        expected_result = ((1.7277246654007437, 1.1415845834992724), -1.2852596484846646)

        result = FrequencyAdjustment.quadratic_critical_point(Q, L, C)
        
        @test all(result[1] .≈ expected_result[1])
        @test result[2] ≈ expected_result[2]
    end
    
    # Test FrequencyAdjustment.innerproduct
    @testset "Inner product $(n)D" for n in 1:4
        S = FrequencyAdjustment.FrequencyAdjuster(reshape(Float64[], zeros(Int, n)...); σ=3)
        temp = radial_chirp(dims=fill(S.L, ndims(S.data[:input])), periods=6)
        A = FFTW.fft(temp)
        B = FFTW.rfft(temp)
        AB = FrequencyAdjustment.innerproduct(S, A, B)
        @test AB ≈ sum(temp[:].^2) * length(A)
    end
    
    @testset "Empty input" begin
        S = FrequencyAdjustment.FrequencyAdjuster(Float64[]; σ=3)
        @test S.data[:input] == Float64[]
        @test FrequencyAdjustment.detect_waves!(S, S.data[:input]) == -1
        @test S.data[:q] == Float64[]
        @test S.data[:waves] == Vector{FrequencyAdjustment.Wave{Float64, 1}}[]
    end
    
    @testset "Real DFT to DFT indices $(n)D" for n in 1:4
        S = FrequencyAdjustment.FrequencyAdjuster(reshape(Float64[], zeros(Int, n)...); σ=3)
        @testset "Radial Chirp" begin
            temp = radial_chirp(dims=fill(S.L, ndims(S.data[:input])), periods=6)
            A = FFTW.fft(temp)
            B = FFTW.rfft(temp)
            realDFT_to_DFT_indices = FrequencyAdjustment.generate_DFT_to_realDFT_indices(S.L, Val(n))
            @test all(A[realDFT_to_DFT_indices[:]] .≈ conj.(B[:]))
        end
        @testset "Random planar wave" begin
            freqs = Tuple(rand(n))
            temp = map(CartesianIndices(ntuple(x->S.L, n))) do I
                return cos(sum(I.I .* freqs))
            end
            A = FFTW.fft(temp)
            B = FFTW.rfft(temp)
            realDFT_to_DFT_indices = FrequencyAdjustment.generate_DFT_to_realDFT_indices(S.L, Val(n))
            @test all(A[realDFT_to_DFT_indices[:]] .≈ conj.(B[:]))
        end
    end
    
    @testset "lerp $(n)D" for n in 1:4
        A = reshape(Complex{Float64}[1:2^n...], ntuple(x->2,n))
        @assert length(A) == 2^n "$A\n does not have a power of 2 length."
        @test 1 == FrequencyAdjustment.lerp_abs(A, ntuple(x->0.0, n)...)
        @test 0.5^n*sum(A) ≈ FrequencyAdjustment.lerp_abs(A, ntuple(x->0.5, n)...)
        @test 2^n == FrequencyAdjustment.lerp_abs(A, ntuple(x->1.0, n)...)
    end
    
    @testset "Local Wave Detection $(n)D" for n in 1:4
        freqs = fill(0.3, n)
        amplitude = 0.5
        S = FrequencyAdjustment.FrequencyAdjuster(reshape(Float64[], zeros(Int, n)...); σ=3)
        s = map(CartesianIndices(zeros(ntuple(x->S.L, n)...))) do I
            pos = [i for i in I.I]
            return amplitude*cos(2pi * sum(freqs.*pos))+0.5
        end
        
        center = ntuple(x->S.L ÷ 2 + 1, n)
        
        FFT = plan_fft!(Array{Complex{Float64}, n}(undef,size(S.windowNd)))
        RFFT = plan_rfft(S.windowNd)
        IRFFT = inv(RFFT)
        
        params = Dict(
            :FFT     => FFT,
            :RFFT    => RFFT,
            :IRFFT   => IRFFT,
            :buffers => FrequencyAdjustment.alloc_buffers(S),
            :max_waves => S.max_waves
        )
        
        (res_windowed, waves) = FrequencyAdjustment.detect_local_waves(S, s, center; params...)
        @test all(isapprox.(res_windowed, 0.5S.windowNd; atol=1e-4))
        @test length(waves) == 1
        @test all(isapprox.(waves[1].freqs, freqs; atol=1e-4))
        @test amplitude ≈ waves[1].amplitude atol=1e-4
    end
end
