# Solving the dynamical formulation of Regularized Unbalanced Optimal Transport
# Imported from the project 
# "Regularized unbalanced optimal transport as entropy minimization with respect to branching Brownian motion"
# by Aymeric Lavenant and Hugo Lavenant#
# See - Arxiv 2111.01666
#     - https://github.com/HugoLav/RegUnOT



using TimerOutputs
using LinearAlgebra
using SparseArrays

function matrixAverageLinear(nSize)
    # Matrix corresponding to average
    output = spzeros(nSize-1,nSize)
    for i = 1:nSize-1
        output[i,i] = 0.5
        output[i,i+1] = 0.5
    end
    return output
end

function matrixAverageLinearPer(nSize)
    # Matrix corresponding to average with periodic boundary conditions
    output = spzeros(nSize,nSize)
    for i = 1:nSize-1
        output[i,i] = 0.5
        output[i,i+1] = 0.5
    end
    output[end,end] = 0.5
    output[end,1] = 0.5
    return output
end

function matrixDerivativeLinear(nSize,delta)
    # Matrix corresponding to derivative in time
    output = spzeros( nSize-1, nSize )
    for i = 1:nSize-1
        output[i,i+1] = 1/delta
        output[i,i] = - 1/delta
    end
    return output
end

function matrixDerivativeLinearPer(nSize,delta)
    # Matrix corresponding to derivative with periodic boundary conditions
    output = spzeros( nSize, nSize )
    for i = 1:nSize-1
        output[i,i+1] = 1/delta
        output[i,i] = - 1/delta
    end
    output[end,end] = -1/delta
    output[end,1] = 1/delta
    return output
end

function matrixLaplacianLinearPer(nSize,delta)
    # Matrix corresponding to Laplacian with periodic boundary conditions
    output = spzeros( nSize, nSize )
    for i = 1:nSize
        output[i,i] = - 2
        if i <= nSize - 1
            output[i,i+1] = 1
        else
            output[i,1] = 1
        end
        if i >= 2
            output[i,i-1] = 1
        else
            output[i,end] = 1
        end
    end
    return 1/delta^2 * output
end


function constructFullMatrix(A,B)
    # If A is a matrix which acts on R^nTime and B a matrix which acts on R^nSpace,
    # returns a matrix which acts on R^{nTime x nSpace} with the right ordering

    rowA = size(A)[1]
    colA = size(A)[2]
    rowB = size(B)[1]
    colB = size(B)[2]

    # Define the full matrix
    output = spzeros(rowA*rowB,colA*colB)

    # Linear and cartesian indices
    linearR = LinearIndices((1:rowA, 1:rowB))
    linearC = LinearIndices((1:colA, 1:colB))

    # Fill the matrix
    for indexA in findall(!iszero, A),  indexB in findall(!iszero, B)
        output[linearR[indexA[1],indexB[1]],linearC[indexA[2],indexB[2]]] = A[indexA]*B[indexB]
    end

    return output
end


function constructFullVector(A,B)
    # If A in R^nTime and B in R^nSpace, return the vector A[t] B[x] with the right ordering

    lengthA = size(A)[1]
    lengthB = size(B)[1]

    # Define the full matrix
    output = zeros(lengthA*lengthB)

    # Linear and cartesian indices
    linearIndex = LinearIndices((1:lengthA, 1:lengthB))

    # Fill the vector
    for i = 1:lengthA, j = 1:lengthB,
        output[linearIndex[i,j]] = A[i]*B[j]
    end

    return output
end



# Big function to build the matrices

function constructMatrices(simP)

    # ------------------------------------------------------------
    # Extract values from the dictionary
    # ------------------------------------------------------------

    nTime = simP["nTime"]
    nSpace = simP["nSpace"]
    deltaT = simP["deltaT"]
    deltaX = simP["deltaX"]
    isGrowth = simP["isGrowth"]
    nu = simP["nu"]
    regInversion = simP["regInversion"]


    # ------------------------------------------------------------
    # Defining auxiliary matrices
    # ------------------------------------------------------------

    # In time
    mDerivativeTime = matrixDerivativeLinear(nTime+1,deltaT)
    mAverageTime = matrixAverageLinear(nTime+1)

    mSelectBoundaryTime = spzeros(2,nTime+1)
    mSelectBoundaryTime[1,1] = 1
    mSelectBoundaryTime[2,end] = 1

    # In space
    mDerivativeSpace = matrixDerivativeLinearPer(nSpace,deltaX)
    mAverageSpace = matrixAverageLinearPer(nSpace)
    mLaplacianSpace = matrixLaplacianLinearPer(nSpace,deltaX)

    # Full matrices acting both on time and space
    mAverageTimeFull = constructFullMatrix(mAverageTime,sparse(I,nSpace,nSpace))
    mDerivativeTimeFull = constructFullMatrix(mDerivativeTime,sparse(I,nSpace,nSpace))

    mAverageSpaceFull = constructFullMatrix(sparse(I,nTime,nTime),mAverageSpace)
    mDerivativeSpaceFull = constructFullMatrix(sparse(I,nTime,nTime),mDerivativeSpace)
    # For the Laplacian, both Laplacian in space and average in time
    mLaplacianSpaceFull = constructFullMatrix(mAverageTime,mLaplacianSpace)

    mBoundaryConditionsFull = constructFullMatrix(mSelectBoundaryTime,sparse(I,nSpace,nSpace))


    # ------------------------------------------------------------
    # Defining the big matrix for the continuity equation and boundary conditions
    # ------------------------------------------------------------

    nRowBigMatrix = nTime*nSpace+2*nSpace
    if isGrowth
        nColumnBigMatrix = (nTime+1)*nSpace+nTime*nSpace+nTime*nSpace
    else
        nColumnBigMatrix = (nTime+1)*nSpace+nTime*nSpace
    end

    bigMatrixLC = spzeros( nRowBigMatrix, nColumnBigMatrix )

    # Derivatives of rho
    if nu > 1e-10
        # d rho/dt - nu/2 Laplacian rho
        bigMatrixLC[ 1:nTime*nSpace, 1:(nTime+1)*nSpace  ] = mDerivativeTimeFull - 0.5*nu*mLaplacianSpaceFull
    else
        # Only the temporal derivative
        bigMatrixLC[ 1:nTime*nSpace, 1:(nTime+1)*nSpace  ] = mDerivativeTimeFull
    end

    # Derivatives of momentum
    bigMatrixLC[ 1:nTime*nSpace, ((nTime+1)*nSpace+1):((nTime+1)*nSpace+nTime*nSpace) ] = mDerivativeSpaceFull

    # Growth part
    if isGrowth
        # For the growth it's the identity
        bigMatrixLC[ 1:nTime*nSpace, ((nTime+1)*nSpace+nTime*nSpace+1):end ] = sparse(-I, nTime*nSpace, nTime*nSpace)
    end

    # Boundary conditions
    bigMatrixLC[ (nTime*nSpace+1):end, 1:(nTime+1)*nSpace ] = mBoundaryConditionsFull

    # Rhs of the continuity equation
    rhs = zeros(nRowBigMatrix)

    # Using this rhsAux to reshuffle the rho0, rho1 to adjust for the BC
    rhsAux = zeros(2,nSpace)
    rhsAux[1,:] = simP["rho0"]
    rhsAux[2,:] = simP["rho1"]

    rhs[(nTime*nSpace+1):end] = vec(rhsAux)

    # ------------------------------------------------------------
    # Cholesly factorizations
    # ------------------------------------------------------------

    choleskyAvgSpace = cholesky(mAverageSpace*mAverageSpace' + sparse(I,nSpace,nSpace))
    choleskyAvgTime = cholesky(mAverageTime*mAverageTime' + sparse(I,nTime,nTime))
    choleskyBigMatrixLC = cholesky( bigMatrixLC * bigMatrixLC' + regInversion * sparse(I, nRowBigMatrix, nRowBigMatrix ))

    # Store the matrices

    simP["bigMatrixLC"] = bigMatrixLC
    simP["rhs"] = rhs
    simP["mAverageSpace"] = mAverageSpace
    simP["mAverageTime"] = mAverageTime
    simP["choleskyBigMatrixLC"] = choleskyBigMatrixLC
    simP["choleskyAvgSpace"] = choleskyAvgSpace
    simP["choleskyAvgTime"] = choleskyAvgTime
    simP["mLaplacianSpace"] = mLaplacianSpace
    # simP["vSPBigMatrix"] = vSPBigMatrix
    # simP["vSPRhoAvg"] = vSPRhoAvg
    # simP["vSPGrowth"] = vSPGrowth

end

function proximalContinuityEquation( currentRho, currentMomentum, currentGrowth, simP)
    # Compute the projection on the set of rho,momentum, growth that satisfy the continuity equation

    nTime = simP["nTime"]
    nSpace = simP["nSpace"]

    if simP["isGrowth"]

        # Compute the discrepancy
        discrepancy = simP["bigMatrixLC"] *  vcat(vec(currentRho), vec(currentMomentum), vec(currentGrowth)) - simP["rhs"]
        # Invert the system
        discrepancy =  simP["choleskyBigMatrixLC"] \ discrepancy
        # Output
        output = vcat(vec(currentRho), vec(currentMomentum), vec(currentGrowth)) - simP["bigMatrixLC"]' * discrepancy
        # Compute the udpate
        return reshape(output[1:(nTime+1)*nSpace], (nTime+1), nSpace ), reshape(output[((nTime+1)*nSpace+1):((nTime+1)*nSpace+nTime*nSpace)], nTime, nSpace ), reshape(output[((nTime+1)*nSpace+nTime*nSpace+1):end], nTime, nSpace )

    else

        # Compute the discrepancy
        discrepancy = simP["bigMatrixLC"] * vcat(vec(currentRho), vec(currentMomentum)) - simP["rhs"]
        # Invert the system
        discrepancy =  simP["choleskyBigMatrixLC"] \ discrepancy
        # Output
        output = vcat(vec(currentRho), vec(currentMomentum)) - simP["bigMatrixLC"]' * discrepancy
        # Compute the udpate
        return reshape(output[1:(nTime+1)*nSpace], (nTime+1), nSpace ), reshape(output[((nTime+1)*nSpace+1):((nTime+1)*nSpace+nTime*nSpace)], nTime, nSpace ), zeros(nTime,nSpace)

    end

end

function proximalAverage(inputRho, inputMomentum,inputGrowth, inputRhoAvg, inputMomentumAvg, inputGrowthAvg, simP)

    nTime = simP["nTime"]
    nSpace = simP["nSpace"]

    # Comptute the discrepancies
    discrepancyRho = simP["mAverageTime"] * inputRho - inputRhoAvg
    discrepancyMomentum = inputMomentum * simP["mAverageSpace"]' - inputMomentumAvg

    # Invert the linear systems
    discrepancyRho =  simP["choleskyAvgTime"] \ discrepancyRho
    discrepancyMomentum =  (simP["choleskyAvgSpace"] \ discrepancyMomentum')'

    # Compute the outputs
    outputRho = inputRho - simP["mAverageTime"]' * discrepancyRho
    outputRhoAvg = inputRhoAvg + discrepancyRho
    outputMomentum = inputMomentum -  discrepancyMomentum * simP["mAverageSpace"]
    outputMomentumAvg = inputMomentumAvg + discrepancyMomentum

    # Then return the values as the average in growth is straightforward

    return outputRho,
        outputMomentum,
        0.5*(inputGrowth+inputGrowthAvg),
        outputRhoAvg,
        outputMomentumAvg,
        0.5*(inputGrowth+inputGrowthAvg)


end

function proximalEnergyStar(a,b,c,argProxE)

    if argProxE[1]
        f = a + 0.5*b^2 + argProxE[3](c)
    else
        f = a + 0.5*b^2
    end

    if f < argProxE[2]
        return 0.
    else

        # Newton's algorithm

        rGuess = 0.

        counter = 0

        while abs(f) > argProxE[2] && counter <= 100

            if argProxE[1]
                cMod = argProxE[6](c,rGuess)
                f = a - rGuess + 0.5*b^2/(1+rGuess)^2 + argProxE[3]( cMod )
                df = -1 - b^2/(1+rGuess)^3 - argProxE[4](cMod)^2 / ( 1+rGuess*argProxE[5](cMod) )
            else
                f = a - rGuess + 0.5*b^2/(1+rGuess)^2
                df = -1 - b^2/(1+rGuess)^3
            end

            rGuess -= f/df

            counter += 1

        end

        if counter == 101
            println("Warning: Newton fails to converge")
        end

        return rGuess

    end

end

function proximalEnergyVectorized(rho,momentum,growth,simP)

    nTime = simP["nTime"]
    nSpace = simP["nSpace"]
    gamma = simP["gamma"]

    outputProj = zeros(nTime,nSpace)

    if simP["isGrowth"]
        outputProj .= proximalEnergyStar.(rho/gamma,momentum/gamma,growth/gamma,Ref(simP["argProxE"]))
    else
        outputProj .= proximalEnergyStar.(rho/gamma,momentum/gamma,0.,Ref(simP["argProxE"]))
    end

    if simP["isGrowth"]
        return gamma*outputProj,
            outputProj .* momentum ./ (1 .+ outputProj),
            growth - gamma * simP["proxPsiStar"].(growth/gamma,outputProj)
    else
        return gamma*outputProj,
            outputProj .* momentum ./ (1 .+ outputProj),
            zeros(nTime,nSpace)
    end

end


function dynamical_RUOT(
    nTime,
    nSpace,
    # Boundary conditions
    rho0,
    rho1;
    # Noise level
    nu::Float64 = 0.,
    # Presence of growth and if yes, Psi* and its prox given as functions
    isGrowth = false,
    # Legendre transform of the growth penalization
    psiStar::Any=nothing,
    # first and second derivative of psiStar
    dPsiStar::Any=nothing,
    d2PsiStar::Any=nothing,
    # Function f(s,gamma) = prox_{gamma*psiStar}(s)
    proxPsiStar::Any=nothing,
    # Number of iterations in Douglas Rachford
    nIter::Int64=1000,
    # Parameters for Douglas Rachford
    alpha = 1.,
    gamma = 1e-3,
    # Tolerance and small parameters
    tolEnergy = 1e-10,
    tolDichotomy = 1e-10,
    maxIterDichotomy = 50,
    regInversion = 1e-10,
    # Displaying the time
    verbose = false
    )

    # Create the timer
    timer = TimerOutput()


    # ------------------------------------------------------------
    # Getting some information out of the arguments
    # ------------------------------------------------------------

    @timeit timer "Precomputations" begin

    # ------------------------------------------------------------
    # Defining some preliminary quantities
    # ------------------------------------------------------------

    gridTimeLarge = LinRange(0,1,nTime+1)
    deltaT = gridTimeLarge[2] - gridTimeLarge[1]
    gridTimeSmall = LinRange(0.5 * deltaT,1 - 0.5 * deltaT,nTime)


    gridSpaceStaggerred = LinRange(0,1-1/nSpace,nSpace)
    deltaX = gridSpaceStaggerred[2] - gridSpaceStaggerred[1]
    gridSpaceCentered = gridSpaceStaggerred .+ deltaX/2


    # ------------------------------------------------------------
    # Define big dictionary with all the fixed variables of the loop
    # ------------------------------------------------------------

    # Create a tuple with the arguments to be passed in prox of the energy

    argProxE = (isGrowth, tolDichotomy,psiStar,dPsiStar,d2PsiStar,proxPsiStar)

    simP = Dict([("nu",nu),
        ("nIter",nIter),
        ("alpha",alpha),
        ("gamma",gamma),
        ("tolDichotomy",tolDichotomy),
        ("maxIterDichotomy",maxIterDichotomy),
        ("isGrowth",isGrowth),
        ("nTime",nTime),
        ("nSpace",nSpace),
        ("deltaT",deltaT),
        ("deltaX",deltaX),
        ("rho0",rho0),
        ("rho1",rho1),
        ("psiStar",psiStar),
        ("dPsiStar",dPsiStar),
        ("d2PsiStar",d2PsiStar),
        ("proxPsiStar",proxPsiStar),
        ("argProxE",argProxE),
        ("regInversion",regInversion)
        ])


    # ------------------------------------------------------------
    # Build the relevant derivation and averaging matrices
    # ------------------------------------------------------------

    constructMatrices(simP)


    # ------------------------------------------------------------
    # Initialize the objects
    # ------------------------------------------------------------

    # The first component indicates if it is either the "z" variable or the "w" variabe

    rho = ones(2, nTime+1,nSpace ) / nSpace
    momentum = zeros(2, nTime, nSpace )
    growth = zeros(2,nTime,nSpace)
    rhoAvg = ones(2,nTime,nSpace) / nSpace
    momentumAvg = zeros(2, nTime, nSpace )
    growthAvg = zeros(2,nTime,nSpace)

    # End of the timer for initialization
    end


    # ------------------------------------------------------------
    # Douglas Rachford loop
    # ------------------------------------------------------------

    @timeit timer "Loop" begin

    # Main loop
    for i = 1:nIter

        # Last udpate: store the values to compute an error
        if i == nIter
            rhoC = copy(rho)
            momentumC = copy(momentum)
            growthC = copy(growth)
            rhoAvgC = copy(rhoAvg)
            momentumAvgC = copy(momentumAvg)
            growthAvgC = copy(growthAvg)
        end


        # First invert the Laplacian matrix
        @timeit timer "Prox CE" auxR, auxM, auxG = proximalContinuityEquation( 2 * rho[1,:,:] - rho[2,:,:], 2 * momentum[1,:,:] - momentum[2,:,:], 2 * growth[1,:,:] - growth[2,:,:], simP)

        # Then compute the proximal map of the energy
        @timeit timer "Prox Energy" auxRBis, auxMBis, auxGBis = proximalEnergyVectorized(2*rhoAvg[1,:,:] - rhoAvg[2,:,:],  2*momentumAvg[1,:,:] - momentumAvg[2,:,:], 2*growthAvg[1,:,:] - growthAvg[2,:,:], simP)

        # Update the w variables
        rho[2,:,:] += alpha * ( auxR - rho[1,:,:]   )
        momentum[2,:,:] += alpha * ( auxM - momentum[1,:,:]   )
        rhoAvg[2,:,:] += alpha * ( auxRBis - rhoAvg[1,:,:]   )
        momentumAvg[2,:,:] += alpha * ( auxMBis - momentumAvg[1,:,:]   )
        growth[2,:,:] += alpha * ( auxG - growth[1,:,:]   )
        growthAvg[2,:,:] += alpha * ( auxGBis - growthAvg[1,:,:]  )


        # Then update the z variables
        @timeit timer "Prox Average" rho[1,:,:], momentum[1,:,:],growth[1,:,:], rhoAvg[1,:,:], momentumAvg[1,:,:], growthAvg[1,:,:]= proximalAverage(rho[2,:,:], momentum[2,:,:],growth[2,:,:], rhoAvg[2,:,:], momentumAvg[2,:,:], growthAvg[2,:,:], simP)

        # Compute an error
        if i == nIter

            errorZ = (sum(abs.( rho[1,:,:] - rhoC[1,:,:] )) + sum(abs.( momentum[1,:,:] - momentumC[1,:,:] )) + sum(abs.( growth[1,:,:] - growthC[1,:,:] )) + sum(abs.( rhoAvg[1,:,:] - rhoAvgC[1,:,:] )) + sum(abs.( momentumAvg[1,:,:] - momentumAvgC[1,:,:] )) + sum(abs.( growthAvg[1,:,:] - growthAvgC[1,:,:] )))/gamma
            errorW = (sum(abs.( rho[2,:,:] - rhoC[2,:,:] )) + sum(abs.( momentum[2,:,:] - momentumC[2,:,:] )) + sum(abs.( growth[2,:,:] - growthC[2,:,:] )) + sum(abs.( rhoAvg[2,:,:] - rhoAvgC[2,:,:] )) + sum(abs.( momentumAvg[2,:,:] - momentumAvgC[2,:,:] )) + sum(abs.( growthAvg[2,:,:] - growthAvgC[2,:,:] )))/gamma

            # Error continuity equation
            if simP["isGrowth"]
                errorCE = sum(abs.(simP["bigMatrixLC"] *  vcat(vec(rho[1,:,:]), vec(momentum[1,:,:]), vec(growth[1,:,:])) - simP["rhs"]))
            else
                errorCE = sum(abs.(simP["bigMatrixLC"] *  vcat(vec(rho[1,:,:]), vec(momentum[1,:,:])) - simP["rhs"]))
            end

            simP["errorZ"] = errorZ
            simP["errorW"] = errorW
            simP["errorCE"] = errorCE

        end

    end

    # End of the timer
    end

    if verbose
        display(timer)
    end

    # Return the output
    return rhoAvg[1,:,:], momentumAvg[1,:,:], growthAvg[1,:,:], simP


end
