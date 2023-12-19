# Produces the convergence plot for the discretization of Papadakis, Peyré and Oudet
# The output is a text file readable by pfgplot 

using DelimitedFiles
using QuadGK
include("dynamical_RUOT.jl")


# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

# Parameters Douglas Rachford
nIter = 10000
alpha = 1.9
gamma = 1e-2


# To solve the plain optimal transport problem with dynamical_RUOT.
# No diffusivity
nu = 0.0 
# No unbalanced optimal transport 
isGrowth = false

# Number of disretization points in time and space
gridSize = [16,32,64,128,256,512]

# Which test case we are looking act 
# See article 
testCase = 1

# ------------------------------------------------------------
# Exact solution 
# ------------------------------------------------------------

# Parameter w 
if testCase == 1
    w = 0.2
elseif testCase == 2
    w = 0.05
end

# name 
if testCase == 1
    nameFile = "case_1.txt"
elseif testCase == 2
    nameFile = "case_2.txt"
end

function periodize(x)
    # Shift x by an integer in order for it to be between -0.5 and 0.5

    a,b = modf(x)

    if a > 0.5 
        return a - 1.
    elseif a < -0.5
        return a+1.
    else
        return a 
    end

end

function rhoFunct(x,t,testCase)
    # Analytical expression of the function rho

    if testCase == 1
        return max.(w .- abs.(periodize(x .- 0.5))/(1+t), 0.)/((1+t)^2*w^2)
    end

    if testCase == 2
        return 1/(2*w) * Float64( abs.(periodize(x .- 0.5)) - t * (periodize(0.5 - w)) <= w ) .* Float64( abs.(periodize(x .- 0.5)) - t * (periodize(0.5 - w)) >= 0. ) 
    end 
end

function rhoIntegrated(x,deltaX,t,testCase)
    # Integrate the analytical expression to get to projected measure Pi \rho 
    return quadgk( y -> rhoFunct(y,t,testCase), x - deltaX/2, x+deltaX/2, rtol=1e-6)[1]
end

function phiFunction(x,t,testCase)
    # Analytical expression of the function phi

    if testCase == 1
        return (x.-0.5).^2 ./ (2 * (1. + t))
    end

    if testCase == 2
        return abs.(x.-0.5) * (0.5 - w)  .- 0.5*(0.5-w)^2 * t
    end
end

function velocityFunction(x,t,testCase)
    # Analytical expression of the velocity field

    if testCase == 1
        return (x .- 0.5)/(1+t)
    end

    if testCase == 2
        if x >= 0.5
            return 0.5 - w 
        else 
            return -(0.5 - w)
        end
    end 
end

function transportCost(testCase)
    # Analytical expression of the transport cost 

    if testCase == 1
        return w^2/12
    end

    if testCase == 2
        return (0.5 - w)^2/2
    end
end 



# ------------------------------------------------------------
# Loop of solutions 
# ------------------------------------------------------------

# create Array with the different errors which are monitored 
errorsArray = zeros((4,length(gridSize)))

for j in 1:length(gridSize)

    n = gridSize[j]

    gridSpace = LinRange(0,1-1/n,n)
    gridSpaceAvg = LinRange(1/(2*n),1-1/(2*n),n)

    # Initial and final conditions 
    rho0 = rhoIntegrated.(gridSpace,1/n,0.,testCase)
    rho1 = rhoIntegrated.(gridSpace,1/n,1.,testCase)

    # Normalize (should not change)
    rho0 /= sum(rho0)
    rho1 /= sum(rho1)

    # ------------------------------------------------------------
    # Compute the solution
    # ------------------------------------------------------------

    println("-"^20)
    println(string("start: n=", n))

    @time rho, momentum, growth, simP = dynamical_RUOT(n,n,rho0,rho1;
        nu = nu,
        isGrowth = false,
        nIter=nIter,
        alpha = alpha,
        gamma = gamma,
        verbose = false)

    
    # Plot the error made
    
    # L^1 norm between rho exact and rho numerical 
    
    errorL1Rho = 0. 
    for i in 1:n
        rhoTh = rhoIntegrated.(gridSpace,1/n,(i-0.5)/n,testCase)
        errorL1Rho += 1/n * sum(abs.(  rho[i,:] - rhoTh/(1e-8 +sum(rhoTh)) )) 
    end 

    # Compute numerical velocity     
    
    # Remove value of the density smaller than tolRho and the cap the velocity   
    tolRho = 1e-10
    densityNonZero = Float64.( rho .> tolRho )    
    velocity = densityNonZero .* momentum ./ rho
    velocity = min.( max.( -20, velocity ), 20. )

    # L^2 norm between nabla phi excat and v numerical
    
    errorL2gradPhi = 0.
    for i in 1:n

        # Compute phi theoritical and then its discrete gradient 
        phiTh = phiFunction(gridSpace,(i-0.5)/n,testCase)
        # Compute the gradient of phiTh 
        gradPhiTh = zeros(n)
        gradPhiTh[1:n-1] = (phiTh[2:n] - phiTh[1:n-1]) * n
        gradPhiTh[n] = ( phiTh[1] - phiTh[n] )*n

        errorL2gradPhi += 1/n * sum( (  velocity[i,:] .- gradPhiTh ).^2 .* rho[i,:] )
    
    end

    # L^2 norm between v exact and v numerical

    errorL2Velocity = 0.
    for i in 1:n
        velocityTh = velocityFunction.(gridSpaceAvg,(i-0.5)/n,testCase) 
        errorL2Velocity += 1/n * sum( (  velocity[i,:] .- velocityTh ).^2 .* rho[i,:] )  
    end

    # Checking the value of the transport cost 
    valueTransport = 0.5 / n * sum( velocity.^2 .* rho )

    println(string("Error L1 density: ", errorL1Rho))
    println(string("Error L2 velocity: ", errorL2Velocity))
    println(string("Error L2 gradient: ", errorL2gradPhi))
    println(string("Value transport: ", valueTransport ))
    println(string("Error transport: ", abs(valueTransport - transportCost(testCase) )))

    errorsArray[1,j] = abs(valueTransport - transportCost(testCase) )
    errorsArray[2,j] = errorL1Rho
    errorsArray[3,j] = errorL2Velocity
    errorsArray[4,j] = errorL2gradPhi

end


# ------------------------------------------------------------
# Write results  
# ------------------------------------------------------------

# Write the results in a format readable by pfgplot 

fileObj = open(nameFile, "w")

# First line: different errors 

write(fileObj, "h errorValue errorRho errorV errorPhi" )
write(fileObj, "\n")

# Other lines: value of the function in space
for j = 1:length(errorsArray[1,:])
    write(fileObj, string( 1/gridSize[j], " "))
    for k = 1:4
        write(fileObj, string( errorsArray[k,j], " "))
    end
    write(fileObj, "\n")
end

close(fileObj) 


