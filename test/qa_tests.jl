@testitem "Quality Assurance" begin
    using Aqua

    Aqua.test_all(AlmostBlockDiagonals)
end

@testitem "JET Package Test" begin
    using JET

    JET.test_package(AlmostBlockDiagonals, target_defined_modules = true)
end
