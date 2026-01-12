clear;clc;
close all;
% Configuration
authkey='edu.jhu.pha.turbulence.testing-201406'; 
dataset='transition_bl'; 
variable='velocity';
temporal_method='none'; 
spatial_method='lag4';  
spatial_operator ='field'; 
% Time parameters
time_start=100.0; 
delta_t=0.5;
num_timesteps=200;
time_end=time_start+(num_timesteps-1)*delta_t;
option=[time_end,delta_t];
times_plot=time_start:delta_t:time_end;
% Grid parameters
nx1=36;
nx2=35;
ny=113;

n_grid_x=nx1+nx2;
n_grid_y=ny;
n_points_total=n_grid_x*n_grid_y;

x_min=500.0;
x_max1=525.0; 
x_max=550.0;  
y_min=0.0036; 
y_max=7.0;    
z_fixed=120.0;

n_points1=nx1*ny;
x_points1=linspace(x_min,x_max1,nx1);
y_points=linspace(y_min,y_max,ny);
[X_grid1,Y_grid1]=meshgrid(x_points1,y_points); 

points1=zeros(n_points1,3);
k=1;
for j=1:ny
    for i=1:nx1
        points1(k,1)=X_grid1(j,i);
        points1(k,2)=Y_grid1(j,i);
        points1(k,3)=z_fixed;
        k=k+1;
    end
end
% Query 1
result_data1=getData(authkey,dataset,variable,time_start,temporal_method,spatial_method,spatial_operator,points1,option);

n_points2=nx2*ny;
x_points2_all=linspace(x_max1,x_max,nx2+1); 
x_points2_unique=x_points2_all(2:end);         

[X_grid2_unique,Y_grid2_unique]=meshgrid(x_points2_unique,y_points);

points2_unique=zeros(n_points2,3);
k=1;
for j=1:ny 
    for i=1:nx2 
        points2_unique(k,1)=X_grid2_unique(j,i);
        points2_unique(k,2)=Y_grid2_unique(j,i);
        points2_unique(k,3)=z_fixed;
        k=k+1;
    end
end
% Query 2
result_data2=getData(authkey,dataset,variable,time_start,temporal_method,spatial_method,spatial_operator,points2_unique,option);
% Combine
result_data=cat(2,result_data1,result_data2); 
X_grid=[X_grid1 X_grid2_unique]; 
Y_grid=[Y_grid1 Y_grid2_unique]; 
% reshaper
function reshaped_data_4D=unflatten_components(flat_data_3D,ny,nx)
    num_timesteps=size(flat_data_3D,1);
    num_components=size(flat_data_3D,3); 
    reshaped_data_4D=zeros(num_timesteps,ny,nx,num_components); 
    for t=1:num_timesteps
        for c=1:num_components
            current_flat=flat_data_3D(t,:,c);
            grid_2d=zeros(ny,nx);            
            k=1;
            for j=1:ny
                for i=1:nx
                    grid_2d(j,i)=current_flat(k);
                    k=k+1;
                end
            end
            reshaped_data_4D(t,:,:,c)=grid_2d;
        end
    end
end
% Reshape
V_reshaped1=unflatten_components(result_data1,ny,nx1); 
V_reshaped2=unflatten_components(result_data2,ny,nx2); 
V_final=cat(3,V_reshaped1,V_reshaped2);
U_x_reshaped=V_final(:,:,:,1);


%% Data Preprocessing and POD Setup (Training/Test Split)

% 1. Define Split Point
num_timesteps_total = num_timesteps;
split_ratio = 0.8;
T_train = floor(num_timesteps_total * split_ratio); % Training snapshots
T_test = num_timesteps_total - T_train;              % Test snapshots
fprintf('Total snapshots: %d. Training snapshots: %d (%.0f%%). Test snapshots: %d.\n', num_timesteps_total, T_train, split_ratio*100, T_test);

% Reshape the data for matrix operations
U_x_permuted = permute(U_x_reshaped, [3, 2, 1]); 
DataMatrix_Ux = reshape(U_x_permuted, [n_points_total, num_timesteps_total]);

% Define time array
t_dmd = times_plot; % Use the defined times_plot array

% Split DataMatrix_Ux
X_train = DataMatrix_Ux(:, 1:T_train);
X_test = DataMatrix_Ux(:, T_train+1:end);
X_total = DataMatrix_Ux;

% 2. Perform SVD/POD on the TRAINING DATA ONLY
X1_train = X_train(:, 1:end-1);
[U2, S2, V2] = svd(X1_train, 'econ'); % U2 is the POD basis
r_inf = 50;
Phi = U2(:, 1:r_inf); % OpInf basis matrix (Phi)
r = r_inf;


%% Hybrid OpInf: Continuous Training on X_train, Euler Prediction on X_total 

X_full = X_train;
X_full_dot = (X_full(:, 2:end) - X_full(:, 1:end-1)) / delta_t; 

X_full = X_full(:, 1:end-1); 
X_full_dot = X_full_dot(:, 1:end);

% Reduced-order states
U_r = Phi' * X_full; 
U_r_dot = Phi' * X_full_dot; 
N_t = size(U_r, 2); 

%Continuous OpInf Least-Squares System
H_term = zeros(r*r, N_t);
for k = 1:N_t
    u_r_k = U_r(:, k);
    H_term(:, k) = kron(u_r_k, u_r_k); 
end
Z = [U_r; H_term]; 
RHS = U_r_dot;     

% Infer the Continuous Operators
G = RHS * pinv(Z); 
A_r = G(:, 1:r);        % Continuous Linear Operator
H_flat = G(:, r+1:end); % Continuous Flat Quadratic Operator

% Prediction (Forward Euler Time Stepping over ALL snapshots)
N_steps = num_timesteps_total;
u_r0 = Phi' * X_total(:, 1); 
U_pred_dt = zeros(r, N_steps);
U_pred_dt(:, 1) = u_r0; 

for k = 1 : N_steps - 1
    u_current = U_pred_dt(:, k);
    
    % Compute the derivative (du/dt)
    u_kron = kron(u_current, u_current); 
    du_dt = A_r * u_current + H_flat * u_kron;
    
    % Forward Euler Integration Step (Discrete Prediction)
    u_next = u_current + delta_t * du_dt; 
    
    U_pred_dt(:, k+1) = u_next;
end

U_r_predicted = U_pred_dt; 

% Reconstruct Full Field and Reshape
X_reconstructed = Phi * U_r_predicted; 
U_dt_reshaped = zeros(N_steps, n_grid_y, n_grid_x);

for tt = 1:N_steps
    snapshot = X_reconstructed(:, tt); 
    k_idx=1;
    for j=1:n_grid_y
        for i=1:n_grid_x
            U_dt_reshaped(tt,j,i) = snapshot(k_idx);
            k_idx = k_idx+1;
        end
    end
end


%% Error Calculation and Plotting

% Error = ||X_true - X_reconstructed||_F / ||X_true||_F
Error_L2 = zeros(1, N_steps);
X_true_total = DataMatrix_Ux;

for tt = 1:N_steps
    % Numerator: Frobenius norm of the residual vector (full field)
    error_vec = X_true_total(:, tt) - X_reconstructed(:, tt);
    numerator = norm(error_vec, 'fro'); 
    
    % Denominator: Frobenius norm of the true field (normalization factor)
    denominator = norm(X_true_total(:, tt), 'fro'); 
    
    Error_L2(tt) = numerator / denominator;
end

figure(5);
plot(t_dmd, Error_L2, 'LineWidth', 2, 'DisplayName', 'OpInf Prediction Error');
hold on;
% Mark the separation point
plot([t_dmd(T_train) t_dmd(T_train)], [0 max(Error_L2)*1.1], 'r--', 'DisplayName', 'Training/Test Split');
hold off;

title('Relative L2-Norm Error: Training vs. Test Performance', 'Interpreter', 'latex');
xlabel('Time ($t$)', 'Interpreter', 'latex');
ylabel('Relative Error $\frac{||U_{true} - U_{pred}||}{||U_{true}||}$', 'Interpreter', 'latex');
legend('show', 'Location', 'northwest');
grid on; 

text(t_dmd(floor(T_train/2)), max(Error_L2)*1.05, 'Training Region', 'Color', [0 0.5 0], 'FontSize', 10, 'HorizontalAlignment', 'center');
text(t_dmd(T_train + floor(T_test/2)), max(Error_L2)*1.05, 'Test Region', 'Color', [0.8 0 0], 'FontSize', 10, 'HorizontalAlignment', 'center');

%% Animation: DT-OpInf Prediction vs True Data
figure(4);
set(gcf, 'Position', [100 100 1100 500]);
sgtitle('DT-OpInf Prediction vs True DNS Data ($U_x$)','Interpreter', 'latex', 'FontSize', 14);

v_min = min(U_x_reshaped(:));
v_max = max(U_x_reshaped(:));

for tt = 1:num_timesteps
    
    % True Data 
    subplot(2,1,1);
    contourf(X_grid, Y_grid, squeeze(U_x_reshaped(tt,:,:)), 40, 'LineStyle', 'none');
    colormap(gca, 'jet');
    c = colorbar;
    c.Label.String = '$U_x$ Velocity (True)';
    c.Label.Interpreter = 'latex';
    title(sprintf('True Data (t = %.2f)', times_plot(tt)), 'Interpreter', 'latex');
    xlabel('X'); ylabel('Y');
    axis tight; set(gca, 'YDir', 'normal');
    caxis([v_min v_max]);

    % DT-OpInf Predicted Data
    subplot(2,1,2);
    contourf(X_grid, Y_grid, squeeze(U_dt_reshaped(tt,:,:)), 40, 'LineStyle', 'none');
    colormap(gca, 'jet');
    c = colorbar;
    c.Label.String = '$U_x$ Velocity (DT-OpInf)';
    c.Label.Interpreter = 'latex';
    title(sprintf('DT-OpInf Prediction (t = %.2f, r=50)', times_plot(tt)), 'Interpreter', 'latex');
    xlabel('X'); ylabel('Y');
    axis tight; set(gca, 'YDir', 'normal');
    caxis([v_min v_max]);
    
    drawnow;
end
%% Default dataset extraction function given in JHTDB website
function result = getData(authToken, dataset, var_original, timepoint_original, temporal_method_original, spatial_method_original, ...
    spatial_operator_original, points, option)

    import matlab.net.http.*
    
    % Determine the number of points and convert the points array to a string
    numPoints = size(points, 1);
    points_str = join(arrayfun(@(i) sprintf('%.8f\t%.8f\t%.8f', points(i, :)), 1:numPoints, 'UniformOutput', false), newline);

    % Retrieve data through REST web service
    options = HTTPOptions('ConnectTimeout', 1000);
    request = RequestMessage('POST', [], points_str);
    
    functionname = 'GetVariable';
    
    if nargin == 8
        url = ['https://web.idies.jhu.edu/turbulence-svc/values?authToken=', authToken, '&dataset=', dataset,...
                '&function=', functionname, '&var=', var_original, ...
                '&t=', num2str(timepoint_original),  '&sint=', spatial_method_original, '&sop=', spatial_operator_original,...
                '&tint=', temporal_method_original];    
    elseif nargin == 9
          url = ['https://web.idies.jhu.edu/turbulence-svc/values?authToken=', authToken, '&dataset=', dataset,...
                '&function=', functionname, '&var=', var_original, ...
                '&t=', num2str(timepoint_original),  '&sint=', spatial_method_original, '&sop=', spatial_operator_original,...
                '&tint=', temporal_method_original, '&timepoint_end=', num2str(option(1)), '&delta_t=', num2str(option(2))];
    else
        error('Incorrect number of arguments.');
    end

    response = request.send(url, options);
    result = response.Body.Data;
    
    if response.StatusCode ~= matlab.net.http.StatusCode.OK
        if isfield(result, 'description')
            error(['HTTP Error ', char(response.StatusCode), ':', newline, ...
                   strjoin(result.description, newline)]);
        else
            error(['HTTP Error ', char(response.StatusCode), '.']);
        end
    end


    if nargin == 9
        times_plot = timepoint_original:option(2):option(1);
        result = reshapeAndPermute(result, var_original, spatial_operator_original, numPoints, length(times_plot));
    end
end

function result = reshapeAndPermute(data, var_original, spatial_operator_original, numPoints, numTimes)
    switch var_original
        case 'velocity'
            result = reshapeByOperator(data, spatial_operator_original, numPoints, numTimes, [3, 9, 18, 3]);
        case 'vectorpotential'
            result = reshapeByOperator(data, spatial_operator_original, numPoints, numTimes, [3, 9, 18, 3]);            
        case 'magneticfield'
            result = reshapeByOperator(data, spatial_operator_original, numPoints, numTimes, [3, 9, 18, 3]);      
        case 'force'
            result = reshapeByOperator(data, spatial_operator_original, numPoints, numTimes, [3, 9, 18, 3]);                
        case 'pressure'
            result = reshapeByOperator(data, spatial_operator_original, numPoints, numTimes, [1, 3, 6, NaN]);
        case 'soiltemperature'
            result = reshapeByOperator(data, spatial_operator_original, numPoints, numTimes, [1, 3, 6, NaN]);
        case 'sgsenergy'
            result = reshapeByOperator(data, spatial_operator_original, numPoints, numTimes, [1, 3, 6, NaN]);
        case 'temperature'
            result = reshapeByOperator(data, spatial_operator_original, numPoints, numTimes, [1, 3, 6, NaN]);
        case 'sgsviscosity'
            result = reshapeByOperator(data, spatial_operator_original, numPoints, numTimes, [1, 3, 6, NaN]);
        case 'density'
            result = reshapeByOperator(data, spatial_operator_original, numPoints, numTimes, [1, 3, 6, NaN]);           
        
        case 'position'
            if strcmp(spatial_operator_original, 'field')
                result = data;
            else
                handleErrorStruct(data, 'Invalid spatial operator for position query.');
            end
        otherwise
                handleErrorStruct(data, ['Unknown variable: ', var_original]);
       end
end

function result = reshapeByOperator(data, operator, numPoints, numTimes, dims)
    switch operator
        case 'field'
            result = reshape(data, [numPoints, numTimes, dims(1)]);
        case 'gradient'
            result = reshape(data, [numPoints, numTimes, dims(2)]);
        case 'hessian'
            result = reshape(data, [numPoints, numTimes, dims(3)]);
        case 'laplacian'
            if isnan(dims(4))
                error(['Laplacian not supported for this variable.']);
            end
            result = reshape(data, [numPoints, numTimes, dims(4)]);
        otherwise
            handleErrorStruct(data, ['Unknown spatial operator: ', operator]);
    end
    result = permute(result, [2, 1, 3]);  % [time, point, component]
end

function handleErrorStruct(data, msg)
    if isstruct(data) && isfield(data, 'description')
        error(['%s', newline, '%s'], msg, strjoin(data.description, newline));
    else
        error(msg);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Getdata Demo notebook (MATLAB)
% 
% supported datasets :
% 
%         - isotropic1024coarse  :  isotropic 1024-cube (coarse).
%         - isotropic1024fine       :  isotropic 1024-cube (fine).
%         - isotropic4096            :  isotropic 4096-cube.
%         - isotropic8192            :  isotropic 8192-cube.
%         - isotropic32768          :  isotropic 32768-cube.
%         - sabl2048low              :  stable atmospheric boundary layer 2048-cube, low-rate timestep.
%         - sabl2048high             :  stable atmospheric boundary layer 2048-cube, high-rate timestep.
%         - stsabl2048low           :  strong stable atmospheric boundary layer 2048-cube, low-rate timestep.
%         - stsabl2048high          :  strong stable atmospheric boundary layer 2048-cube, high-rate timestep.
%         - rotstrat4096               :  rotating stratified 4096-cube.
%         - mhd1024                   :  magneto-hydrodynamic isotropic 1024-cube.
%         - mixing                       :   homogeneous buoyancy driven 1024-cube.
%         - channel                     :  channel flow.
%         - channel5200              :  channel flow (reynolds number 5200).
%         - transition_bl               :  transitional boundary layer.
% functions :
% 
%         - getData  :  retrieve (interpolate and/or differentiate) field data on a set of specified spatial points for the specified variable.        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% instantiate dataset 
%
% purpose :
%        - instantiate the dataset and cache the metadata.
%
% parameters :
% 
%        - auth_token    :  turbulence user authorization token.
%        - dataset_title  :  name of the turbulence dataset.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 

% ---- Enter user JHTDB token ----
authkey = 'edu.jhu.pha.turbulence.testing-201406';  
% the above is a default testing token that works for queries up to 4096 points
% for larger queries, please request token at Please send an e-mail to 
% turbulence@lists.johnshopkins.edu including your name, email address, 
% and institutional affiliation and department, together with a short 
% description of your intended use of the database.
%
% ---- select dataset ----

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% getData 
%
% purpose :
%        - retrieve (interpolate and/or differentiate) a group of sparse data points.
%
% steps :
% 
%          - step 1  :  identify the database files to be read.
%          - step 2  :  read the database files and store the interpolated points in an array.
% 
% parameters :
% 
%          - dataset  :  the instantiated dataset.
%          - points  :  array of points in the domain [0, 2pi).
%          - variable  :  type of data (velocity, pressure, energy, temperature, force, magneticfield, vectorpotential, density, position).
%          - time  :  time point (snapshot number for datasets without a full time evolution).
%          - time_end  :  ending time point for 'position' variable and time series queries.
%          - delta_t  :  time step for 'position' variable and time series queries.
%          - temporal_method  :  temporal interpolation methods.
%                 - none  :  No temporal interpolation (the value at the closest stored time will be returned).
%                 - pchip  :  Piecewise Cubic Hermite Interpolation Polynomial method is used, in which the value from the two nearest time points
%                    is interpolated at time t using Cubic Hermite Interpolation Polynomial, with centered finite difference evaluation of the
%                    end-point time derivatives (i.e. a total of four temporal points are used).
%          - spatial_method  :  spatial interpolation and differentiation methods.
%                 - none      :  No spatial interpolation (value at the datapoint closest to each coordinate value).
%                 - lag4       :  4th-order Lagrange Polynomial interpolation along each spatial direction.
%                 - lag6       :  6th-order Lagrange Polynomial interpolation along each spatial direction.
%                 - lag8       :  8th-order Lagrange Polynomial interpolation along each spatial direction.
%                 - m1q4     :  Splines with smoothness 1 (3rd order) over 4 data points.
%                 - m2q8     :  Splines with smoothness 2 (5th order) over 8 data points.
%                 - m2q14    :  Splines with smoothness 2 (5th order) over 14 data points.
%                 - fd4noint  :  4th-order centered finite differencing (without spatial interpolation).
%                 - fd6noint  :  6th-order centered finite differencing (without spatial interpolation).
%                 - fd8noint  :  8th-order centered finite differencing (without spatial interpolation).
%                 - fd4lag4   :  4th-order Lagrange Polynomial interpolation in each direction, of the 4th-order finite difference values on the grid.
%           - spatial_operator  :  spatial interpolation and differentiation operator.
%                 - field         :  function evaluation & interpolation.
%                 - gradient   :  differentiation & interpolation.
%                 - hessian    :  differentiation & interpolation.
%                 - laplacian   :  differentiation & interpolation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%