%% RANS X-Momentum SINDy for UNSTEADY FLOW
clear; clc; close all;

% Constants and Configuration 
authkey='edu.jhu.pha.turbulence.testing-201406'; 
dataset='transition_bl'; 
spatial_op = 'field';
temporal_method='none'; 
spatial_method = 'none'; % Required for getData

rho = 1.293;     % Density (incompressible)
nu = 1.25e-3;   % Kinematic viscosity

% Grid Setup (Reduced for Stability) 
nx1=30; nx2=30; ny=67;
n_grid_x=nx1+nx2; n_grid_y=ny;
n_points_total=n_grid_x*n_grid_y; % 400 total points

x_points1=linspace(500.0, 515.0, nx1);
x_points2_all=linspace(515.0, 530.0, nx2+1); x_points2_unique=x_points2_all(2:end);
x_points_all=[x_points1 x_points2_unique];
y_points=linspace(0.0036, 4.0, ny);
z_fixed=120.0;

%  Time parameters 
time_start=100.0; 
delta_t=0.3;
num_timesteps=5; % Using 5 steps for better Ruv averaging; central diff is used for 3 steps (t2, t3, t4)
times_to_query = time_start + (0:num_timesteps-1) * delta_t; 

points_full=zeros(n_points_total,3);
k_idx=1;
for j=1:ny
    for i=1:n_grid_x
        points_full(k_idx,1)=x_points_all(i);
        points_full(k_idx,2)=y_points(j);
        points_full(k_idx,3)=z_fixed;
        k_idx=k_idx+1;
    end
end


% Data Retrieval (U, V, and P)
U_data_inst = zeros(num_timesteps, n_grid_y, n_grid_x, 1);
V_data_inst = zeros(num_timesteps, n_grid_y, n_grid_x, 1);
P_data_inst = zeros(num_timesteps, n_grid_y, n_grid_x, 1);

for t_idx = 1:num_timesteps
    t_point = times_to_query(t_idx);
    
    % Query 1: Velocity (u, v, w)
    result_vel = getData(authkey, dataset, 'velocity', t_point, temporal_method, spatial_method, spatial_op, points_full);
    V_data = unflatten_components(result_vel, ny, n_grid_x);
    U_data_inst(t_idx, :, :, 1) = squeeze(V_data(1, :, :, 1)); % u (x-component)
    V_data_inst(t_idx, :, :, 1) = squeeze(V_data(1, :, :, 2)); % v (y-component)
    
    % Query 2: Pressure (p)
    result_press = getData(authkey, dataset, 'pressure', t_point, temporal_method, spatial_method, spatial_op, points_full);
    P_data = unflatten_components(result_press, ny, n_grid_x);
    P_data_inst(t_idx, :, :, 1) = squeeze(P_data(1, :, :, 1)); % p
end



% Time-Averaged Mean Velocities (U and V) over all samples
U_mean_time_avg = mean(U_data_inst, 1); % Size: [1, n_grid_y, n_grid_x, 1]
V_mean_time_avg = mean(V_data_inst, 1); % Size: [1, n_grid_y, n_grid_x, 1]

% Replicate the [1, Y, X, 1] array num_timesteps times in the first dimension.
U_mean_repmat = repmat(U_mean_time_avg, [num_timesteps, 1, 1, 1]); 
V_mean_repmat = repmat(V_mean_time_avg, [num_timesteps, 1, 1, 1]); 

% Fluctuations: u' = u - U_mean_time_avg
U_prime = U_data_inst - U_mean_repmat;
V_prime = V_data_inst - V_mean_repmat;

% Calculate Reynolds Stress: <u'v'>, 
Ruv_avg_tensor = squeeze(mean(U_prime .* V_prime, 1)); 

LHS_U_vec_all = []; 
Theta_all = [];     

t_start_idx = 2;
t_end_idx = num_timesteps - 1;

for t_idx = t_start_idx : t_end_idx
    U_k = squeeze(U_data_inst(t_idx, :, :));
    V_k = squeeze(V_data_inst(t_idx, :, :));
    P_k = squeeze(P_data_inst(t_idx, :, :));  
    % Data at time k-1 and k+1 for time derivative
    U_k_minus_1 = squeeze(U_data_inst(t_idx - 1, :, :));
    U_k_plus_1 = squeeze(U_data_inst(t_idx + 1, :, :));

    dU_dt = (U_k_plus_1 - U_k_minus_1) / (2 * delta_t);
    
    % Spatial Derivatives for RHS library
    [dU_dx, dU_dy] = gradient(U_k, x_points_all, y_points);
    [~, d2U_dy2] = gradient(dU_dy, x_points_all, y_points);
    [dP_dx, ~] = gradient(P_k, x_points_all, y_points);
    
    %  Reynolds Stress Gradient: d/dy(<u'v'>)
    % Since Ruv_avg_tensor is time-averaged, using it for all time steps
    [~, dRuv_avg_dy] = gradient(Ruv_avg_tensor, x_points_all, y_points);
    
    % Construct LHS and RHS
    LHS_U_k = dU_dt;    
    Theta_k = zeros(n_points_total, 5); 
    
    % Term 1 (T1): Convection 1: -U * dU/dx
    Theta_k(:, 1) = -U_k(:) .* dU_dx(:); 
    
    % Term 2 (T2): Convection 2: -V * dU/dy
    Theta_k(:, 2) = -V_k(:) .* dU_dy(:); 
    
    % Term 3 (T3): Pressure Gradient Term: (-1/rho) * dP/dx
    Theta_k(:, 3) = (-1/rho) * dP_dx(:); 
    
    % Term 4 (T4): Viscous Term: nu * d2U/dy2
    Theta_k(:, 4) = nu * d2U_dy2(:); 
    
    % Term 5 (T5): Reynolds Stress Term: d/dy(<u'v'>)
    Theta_k(:, 5) = dRuv_avg_dy(:); 
    LHS_U_vec_all = [LHS_U_vec_all; LHS_U_k(:)];
    Theta_all = [Theta_all; Theta_k];
end

% Sparse Regression 
lambda = 0.001; 
N_data_points = length(LHS_U_vec_all);

Xi = Theta_all \ LHS_U_vec_all; 

for iter = 1:15
    Xi_old = Xi;
    Xi(abs(Xi) < lambda) = 0;
    active_terms = (Xi ~= 0);
    Theta_reduced = Theta_all(:, active_terms);
    
    if ~isempty(Theta_reduced)
        Xi(active_terms) = Theta_reduced \ LHS_U_vec_all;
    end
    
    if norm(Xi - Xi_old) < 1e-6
        break;
    end
end


term_names = { '-U * dU/dx';'-V * dU/dy'; '(-1/rho) * dP/dx';'nu * d^2U/dy^2'; 'd/dy(<u\v\>)'};

equation_string = 'dU/dt = ';
active_term_count = 0;

for i = 1:5
    coeff = Xi(i);
    term_name = term_names{i};
    
    if abs(coeff) > 1e-4 % Checking if coefficient is significant
        if active_term_count > 0 
            if coeff > 0
                sign_str = ' + ';
            else
                sign_str = ' - ';
            end
        else % First active term
             if coeff < 0
                sign_str = '-';
            else
                sign_str = '';
            end
        end
        
        equation_string = [equation_string sign_str sprintf('%.4f * (%s)', abs(coeff), term_name)];
        active_term_count = active_term_count + 1;
    end
end

fprintf('\nDiscovered Equation:\n%s\n', equation_string);
fprintf('\nCoefficients (Ideally near 1.0):\n');
fprintf('Xi_1 (Convection U*dU/dx Coefficient): %.4f\n', Xi(1));
fprintf('Xi_2 (Convection V*dU/dy Coefficient): %.4f\n', Xi(2));
fprintf('Xi_3 (Pressure Term Coefficient): %.4f\n', Xi(3));
fprintf('Xi_4 (Viscous Term Coefficient): %.4f\n', Xi(4));
fprintf('Xi_5 (Reynolds Stress Term Coefficient): %.4f\n', Xi(5));

% The 'unflatten_components' function
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

% --- The 'getData' function 
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
    
    % Manual Reshaping for single time point (nargin == 8) is required here
    if nargin == 8 
        % Determine the number of components for the given variable
        switch var_original
            case {'velocity', 'vectorpotential', 'magneticfield', 'force'}
                dims = 3; 
            case {'pressure', 'soiltemperature', 'sgsenergy', 'temperature', 'sgsviscosity', 'density'}
                dims = 1;
            case 'reynoldsstress'
                dims = 6; % Symmetric tensor components
            otherwise
                dims = 1; 
        end
        % Manual Reshaping for single time point [time=1, point, component]
        if ~strcmp(var_original, 'position')
            result = reshape(result, [numPoints, 1, dims]); 
            result = permute(result, [2, 1, 3]); % [time, point, component]
        end
    % Original nargin == 9 logic
    elseif nargin == 9
        times_plot = timepoint_original:option(2):option(1);
        result = reshapeAndPermute(result, var_original, spatial_operator_original, numPoints, length(times_plot));
    end
end
% --- The reshapeAndPermute, reshapeByOperator, and handleErrorStruct functions 
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

