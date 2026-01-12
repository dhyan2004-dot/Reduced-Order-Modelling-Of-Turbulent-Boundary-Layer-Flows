clc;clear;close all;

% JHTDB Configuration and Parameters
authkey='edu.jhu.pha.turbulence.testing-201406'; 
dataset='transition_bl'; 
variable='velocity';
temporal_method='none'; 
spatial_method='lag4';  
spatial_operator ='field'; 
% N = 4701.   
num_timesteps=300; 
delta_t=0.25;
time_start = 100.0; 
time_end1 = time_start + (num_timesteps - 1) * delta_t;
time_end =time_end1+(num_timesteps-1)*delta_t;% Final time point
times_plot = time_start : delta_t : time_end1;
times_plot1=time_end1:delta_t:time_end;
option = [time_end1, delta_t];
option1 =[time_end,delta_t];

% data for ONLY 5 distinct points 
n_grid_x = 1;
n_grid_y = 5;
n_points_total = n_grid_x * n_grid_y;
x_fixed = 525.0; 
y_points = [0.0036,0.1,0.5, 3.5, 6.9]; 
z_fixed = 120.0; % Fixed z-coordinate

points_3 = zeros(n_points_total, 3);
points_3(1, :) = [x_fixed, y_points(1), z_fixed]; 
points_3(2, :) = [x_fixed, y_points(2), z_fixed]; 
points_3(3, :) = [x_fixed, y_points(3), z_fixed];
points_3(4, :) = [x_fixed, y_points(4), z_fixed];
points_3(5, :) = [x_fixed, y_points(5), z_fixed];

result_data3 = getData(authkey, dataset, variable, time_start, temporal_method,spatial_method, spatial_operator, points_3, option);
result_data4 = getData(authkey, dataset, variable, time_end1, temporal_method,spatial_method, spatial_operator, points_3, option1);
result_data=cat(1,result_data3,result_data4);

U_x_time_series = squeeze(result_data(:, :, 1)); % Streamwise Velocity (Component 1)

figure(5);
set(gcf,'Position',[100 100 900 600]);
max_lag_time=150; 
max_lag_samples=round(max_lag_time / delta_t);
Point_Names = {'P1: Near Wall (y=0.0036)','P2: Near wall (y=0.1)','P3: Mid-BL (y=0.5)', 'P4: Mid-BL (y=3.5)', 'P5: Free Stream (y=6.9)'};
Colors = {'#0072BD', '#D95319', '#77AC30','#FF00FF','#000000'};
means=[0.00675,0.18790,0.56616,0.85436,0.98090];
for k = 1:size(U_x_time_series, 2)
    U_series = U_x_time_series(:, k);   
    % essential for stationarity/periodicity
    U_fluctuations = U_series - means(k);    
    % Autocorrelation using 'xcorr'
    % 'coeff' normalizes the ACF such that C(0) = 1
    [C, lags_idx] = xcorr(U_fluctuations, max_lag_samples, 'coeff');
    % tau >= 0
    zero_lag_idx = find(lags_idx == 0);
    C_pos = C(zero_lag_idx:end);
    lags_pos = lags_idx(zero_lag_idx:end);
    % lag samples to physical time 
    time_lags = lags_pos * delta_t;
    plot(time_lags, C_pos, 'Color', Colors{k}, 'LineWidth', 2, 'DisplayName', Point_Names{k});
    hold on;
end

yline(0, 'k--', 'LineWidth', 1, 'DisplayName', 'Zero Correlation');
title('Autocorrelation of Streamwise Velocity Fluctuations ($\mathbf{U_x}$)', 'Interpreter', 'latex', 'FontSize', 14);
xlabel('Time Lag ($\tau$) [s]', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('Autocorrelation Coefficient', 'FontSize', 12);
legend('show', 'Location', 'northeast');
grid on;
axis tight;
ylim([-0.5 1.05]); 
xlim([0 max_lag_time]);
hold off;

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