%% *Pinns for Inverse Problems : Constant Coefficients*
%
%
% The linearized one-dimensional shallow water (Saint-Venant) equations for
% a channel with a flat bottom and constant water depth $H$ reduce to the shallow
% water wave equation $$\frac{\partial^2 \eta}{\partial t^2} - gH\frac{\partial^2
% \eta}{\partial x^2} = 0$$ in $\Omega$, $\eta = 0$ at the channel ends, where
% $\eta(x,t)$ is the free-surface elevation and $\Omega = (0,L)\times(0,T)$ is
% the space-time domain. The channel ($L$ = 1 m) opens at both ends into large
% reservoirs held at the mean water level, so $\eta = 0$ at $x = 0$ and $x =
% L$. The exact standing-wave (seiche) solution when $gH = 0.981$ m^2/s^2 (with
% $g = 9.81$ m/s^2 and $H = 0.1$ m) is
%
% $$\eta(x,t)= \sin(\pi x)\cos(\omega t), \qquad \omega = \pi\sqrt{gH}$$
%
% This equation is relevant for example in coastal and river engineering, where
% it describes long waves such as tides, seiches, flood and tsunami waves in
% shallow water; the constant coefficient $gH$ is the squared wave celerity,
% set by the water depth $H$ (the bathymetry). The typical "forward" problem
% is to use this information to find $\eta$.
%
% One could consider the inverse problem: given the surface elevation $\eta$
% measured at a set of gauge points, the known gravitational acceleration $g$,
% and boundary conditions, what is the water depth? This is the classical bathymetry
% inversion problem, where the depth is inferred from the observed wave motion.
% Using a PINN to solve the equation, we can optimize the PINN parameters and
% the optimize to solve for the coefficient at the same time. In this exmaple,
% we will take the data to be given by the exact solution above. In practice,
% the exact solution is often not known, but measurements of the solution can
% be provided instead.
%% PDE Model Definition
%

rng('default'); % for reproducibility
model = createpde();
L = 1; % channel length (m)
T = 2; % measurement time window (s), about one seiche period 2*L/sqrt(gH)
R1 = [3;4;0;L;L;0;0;0;T;T]; % rectangular space-time domain in the (x,t) plane
geometryFromEdges(model,decsg(R1));
%%
% The surface elevation is known on the whole boundary of the space-time domain:
% $\eta = 0$ at the channel ends ($x = 0$ and $x = L$) and the measured elevation
% at the initial and final times ($t = 0$ and $t = T$). These values are taken
% from the data function during training, so no constant Dirichlet condition
% needs to be assigned to the PDE model.
%
% Plot the geometry and display the edge labels. The two vertical edges are
% the channel ends and the two horizontal edges are the initial and final times.

figure
pdegplot(model,"EdgeLabels","on");
axis equal
%%
% Create a structural array of coefficents. Specify the coefficients to the
% PDE model. Note that pdeCoeffs.c is an initial guess for $gH$; it will be updated
% during training. We also define pdeCoeffs to be a struct of dlarrays so that
% we can compute gradients with respect to them.

pdeCoeffs.c = .5; % initial guess for gH (m^2/s^2); the exact value is 0.981
% fixed values for other coefficients
pdeCoeffs.m = 1; % coefficient of the second time derivative
pdeCoeffs.d = 0;
pdeCoeffs.a = 0;
pdeCoeffs.f = 0;
% set up model
specifyCoefficients(model,"m",pdeCoeffs.m,"d",pdeCoeffs.d,"c",pdeCoeffs.c,"a",pdeCoeffs.a,"f",pdeCoeffs.f);
Hmax = 0.05; Hgrad = 2; Hedge = Hmax/10;
msh = generateMesh(model,"Hmax",Hmax,"Hgrad",Hgrad,"Hedge",{1:model.Geometry.NumEdges,Hedge});
% make coefs dlarrays so gradients can be computed
pdeCoeffs = structfun(@dlarray,pdeCoeffs,'UniformOutput',false);
%% Generate Space-Time Data for Training PINN
% This examples uses mesh nodes as the collocation points. Model loss at the
% collocation points on the domain and boundary are used to train the PINN.

boundaryNodes = findNodes(msh,"region","Edge",1:model.Geometry.NumEdges);
domainNodes = setdiff(1:size(msh.Nodes,2),boundaryNodes);
domainCollocationPoints = msh.Nodes(:,domainNodes)';
%% Define Deep Learning Model
% This is a neural network with 3 hidden layers and 50 neurons per layer. The
% two inputs to the network correspond to the x coordinate and the time t, and
% the one output corresponds to the surface elevation, so |predict(pinn,XT)|
% appoximates |eta(x,t)|. While training the gH coefficient, we will also train
% this neural network to provide solutions to the PDE.

numLayers = 3;
numNeurons = 50;
layers = featureInputLayer(2);
for i = 1:numLayers-1
    layers = [
        layers
        fullyConnectedLayer(numNeurons)
        tanhLayer];%#ok<AGROW>
end
layers = [
    layers
    fullyConnectedLayer(1)];
pinn = dlnetwork(layers);
%% Define Custom Training Loop to Train the PINN Using ADAM Solver
% Specify training options. Create arrays for average gradients and square gradients
% for both the PINN and the parameter; both will be trained using the ADAM solver.

numEpochs = 1500;
miniBatchSize = 2^12;
initialLearnRate = 0.01;
learnRateDecay = 0.001;
averageGrad = []; % for pinn updates
averageSqGrad = [];
pAverageGrad = []; % for parameter updates
pAverageSqGrad = [];
%%
% Setup data store for the training points. For simplicity, we both train the
% PINN and compute the known data at the mesh nodes.

ds = arrayDatastore(domainCollocationPoints);
mbq = minibatchqueue(ds, MiniBatchSize = miniBatchSize, MiniBatchFormat="BC");
%%
% Calculate the total number of iterations for the training progress monitor
% and initialize the monitor.

numIterations = numEpochs * ceil(size(domainCollocationPoints,1)/miniBatchSize);
monitor = trainingProgressMonitor(Metrics="Loss",Info="Epoch",XLabel="Iteration");
%% Training Loop
% Train the model and parameter using a custom training loop. Update the network
% parameters using the adamupdate function. At the end of each iteration, display
% the training progress. Note that we allow the PINN to be trained for 1/10th
% of the epochs before updating the gH coefficient. This helps with robustness
% to the initial guess.

iteration = 0;
epoch = 0;
learningRate = initialLearnRate;
lossFcn = dlaccelerate(@modelLoss);
while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;
    reset(mbq);
    while hasdata(mbq) && ~monitor.Stop
        iteration = iteration + 1;
        XT = next(mbq);
        % Evaluate the model loss and gradients using dlfeval.
        [loss,gradients] = dlfeval(lossFcn,model,pinn,XT,pdeCoeffs);
        % Update the network parameters using the adamupdate function.
        [pinn,averageGrad,averageSqGrad] = adamupdate(pinn,gradients{1},averageGrad,...
                                               averageSqGrad,iteration,learningRate);
        % Update the gH coefficient using the adamupdate function. Defer
        % updating until 1/10 of epochs are finished.
        if epoch > numEpochs/10
            [pdeCoeffs.c,pAverageGrad,pAverageSqGrad] = adamupdate(pdeCoeffs.c,gradients{2},pAverageGrad,...
                                               pAverageSqGrad,iteration,learningRate);
        end
    end
    % Update learning rate.
    learningRate = initialLearnRate / (1+learnRateDecay*iteration);
    % Update the training progress monitor.
    recordMetrics(monitor,iteration,Loss=loss);
    updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
    monitor.Progress = 100 * iteration/numIterations;
end
%% Visualize Data
% Evaluate PINN at the space-time mesh nodes and plot, include updated value
% of gH and the recovered water depth H = gH/g in the title. A typical run recovers
% gH = 0.957, i.e. a depth H = 0.0976 m compared to the exact value of 0.1 m,
% a 2.4% error; training for more epochs reduces the error further.

nodesDLarry = dlarray(msh.Nodes,"CB");
Upinn = gather(extractdata(predict(pinn,nodesDLarry)));
figure;
pdeplot(model,"XYData",Upinn);
xlabel("x (m)");
ylabel("t (s)");
title(sprintf("Solution with gH = %.4f, recovered depth H = %.4f m", ...
    double(pdeCoeffs.c),double(pdeCoeffs.c)/9.81));
%%
%
%
%
%% Model Loss Function
% The |modelLoss| helper function takes a |dlnetwork| object |pinn| and a mini-batch
% of input data |XT|, and returns the loss and the gradients of the loss with
% respect to the learnable parameters in |pinn| and with respect to the gH coefficient.
% To compute the gradients automatically, use the |dlgradient| function. Return
% the gradients w.r.t. learnable and w.r.t the parameter as two elements of a
% cell array so they can be used separately.  The model is trained by enforcing
% that given an input $(x,t)$ the output of the network $\eta(x,t)$ satsifies
% the shallow water wave equation, the measured data, and the boundary conditions.

function [loss,gradients] = modelLoss(model,pinn,XT,pdeCoeffs)
U = forward(pinn,XT);

% Loss for difference in data taken at mesh nodes.
Utrue = getSolutionData(XT);
lossData = l2loss(U,Utrue);

% Compute gradients of U w.r.t. x and t, and the second derivatives.
gradU = dlgradient(sum(U,"all"),XT,EnableHigherDerivatives=true);
gradUxx = dlgradient(sum(pdeCoeffs.c.*gradU(1,:),"all"),XT,EnableHigherDerivatives=true);
gradUtt = dlgradient(sum(gradU(2,:),"all"),XT,EnableHigherDerivatives=true);

% Enforce PDE. Calculate lossF from the residual of the shallow water wave
% equation m*eta_tt - gH*eta_xx + a*eta = f.
res = pdeCoeffs.m.*gradUtt(2,:) - gradUxx(1,:) + pdeCoeffs.a.*U - pdeCoeffs.f;
lossF = mean(sum(res.^2,1),2);

% Enforce boundary and initial conditions. Calculate lossU. The surface
% elevation on the space-time boundary (channel ends and initial/final
% times) is known from the data.
BC_XT = []; % boundary coordinates
% Loop over the boundary edges and find boundary coordinates.
numBoundaries = model.Geometry.NumEdges;
for i=1:numBoundaries
    BCiNodes = findNodes(model.Mesh,"region","Edge",i);
    BC_XT = [BC_XT, model.Mesh.Nodes(:,BCiNodes)]; %#ok<AGROW>
end
BC_XT = dlarray(BC_XT,"CB"); % format the coordinates
actualBC = getSolutionData(BC_XT); % contains the actual boundary information
predictedBC = forward(pinn,BC_XT);
lossBC = mse(predictedBC,actualBC);

% Combine weighted losses.
lambdaPDE  = 0.4; % weighting factor
lambdaBC   = 0.6;
lambdaData = 0.5;
loss = lambdaPDE*lossF + lambdaBC*lossBC + lambdaData*lossData;

% Calculate gradients with respect to the learnable parameters and
% gH-coefficient. Pass back cell array to update pinn and coef separately.
gradients = dlgradient(loss,{pinn.Learnables,pdeCoeffs.c});
end
%%
%
%
%
%% Data Function
% This function returns the solution data at a given set of points |XT|. As
% a demonstration of the method, we return the exact solution from this function,
% but this function could be replaced with measured data for a given application.
% The surface elevation is normalized by the wave amplitude; the equation is
% linear and homogeneous, so this scaling does not affect the recovered coefficient.

function UD = getSolutionData(XT)
    omega = pi*sqrt(9.81*0.1); % dispersion relation omega = pi*sqrt(gH) with the exact depth H = 0.1 m
    UD = sin(pi*XT(1,:)).*cos(omega*XT(2,:));
end
%%
% Adapted from the MathWorks example "PINNs for Inverse Problems: Constant
% Coefficients" (Poisson equation on a unit disk). Copyright 2023 The MathWorks, Inc.
