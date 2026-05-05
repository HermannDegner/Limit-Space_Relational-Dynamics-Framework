



F = M * EFP
E = F_next - M * F
Hvec += dt * (E - A * Hvec)
H = norm(Hvec)
Dξ = K / norm(M)
ξ = Dξ * randn(n)
M += dt * η * (E + ξ) * F'

if H > θ + norm(ξ)
    M += 0.1U
    Hvec *= ρ
end