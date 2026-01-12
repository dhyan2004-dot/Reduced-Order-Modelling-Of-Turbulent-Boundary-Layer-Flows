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
%%  DMD Setup 

% Using the same 80% split
num_timesteps_total = num_timesteps;
split_ratio = 0.9;
T_train = floor(num_timesteps_total * split_ratio); % Training snapshots
T_test = num_timesteps_total - T_train;              % Test snapshots
U_x_permuted = permute(U_x_reshaped, [3, 2, 1]); 
DataMatrix_Ux = reshape(U_x_permuted, [n_points_total, num_timesteps_total]);
X_total = DataMatrix_Ux;

t_total = 0:delta_t:(num_timesteps_total-1)*delta_t;
t_train = t_total(1:T_train);
% split
X_train = X_total(:, 1:T_train);
X1_train = X_train(:, 1:end-1); 
X2_train = X_train(:, 2:end);   
%% DMD Training Only
r_dmd = 50; 
r = r_dmd;
[U2, S2, V2] = svd(X1_train, 'econ');
U_r = U2(:, 1:r);
S_r = S2(1:r, 1:r);
V_r = V2(:, 1:r);

A_tilde = U_r' * X2_train * V_r / S_r;
[W_r, D_r] = eig(A_tilde);

Phi_dmd = X2_train * V_r / S_r * W_r;

lambda = diag(D_r);
omega = log(lambda) / delta_t;

x1_train = X1_train(:, 1);
b = Phi_dmd \ x1_train;


%% DMD Prediction 

time_dynamics = zeros(length(b), length(t_total));
for tt = 1:length(t_total)
    time_dynamics(:, tt)=(b.*exp(omega*t_total(tt)));
end

X_dmd_pred = real(Phi_dmd * time_dynamics);


%% Error Calculation and Plotting

% Error = ||X_true - X_pred||_F / ||X_true||_F
N_steps = num_timesteps_total;
Error_L2_DMD = zeros(1, N_steps);

for tt = 1:N_steps
    error_vec = X_total(:, tt) - X_dmd_pred(:, tt);
    numerator = norm(error_vec, 'fro'); 
    denominator = norm(X_total(:, tt), 'fro');    
    Error_L2_DMD(tt) = numerator / denominator;
end

figure(); 
plot(t_total, Error_L2_DMD, 'LineWidth', 2, 'DisplayName', 'DMD Prediction Error');
hold on;
plot([t_total(T_train) t_total(T_train)], [0 max(Error_L2_DMD)*1.1], 'r--', 'DisplayName', 'Training/Test Split');
hold off;

title('DMD: Relative L2-Norm Error: Training vs. Test Performance', 'Interpreter', 'latex');
xlabel('Time ($t$)', 'Interpreter', 'latex');
ylabel('Relative Error $\frac{||U_{true} - U_{pred}||}{||U_{true}||}$', 'Interpreter', 'latex');
legend('show', 'Location', 'northwest');
grid on;
set(gca, 'YScale', 'log'); 

T_test_start_index = T_train + floor(T_test/2);
text(t_total(floor(T_train/2)), max(Error_L2_DMD)*1.05, 'Training Region', 'Color', [0 0.5 0], 'FontSize', 10, 'HorizontalAlignment', 'center');
text(t_total(T_test_start_index), max(Error_L2_DMD)*1.05, 'Test/Forecast Region', 'Color', [0.8 0 0], 'FontSize', 10, 'HorizontalAlignment', 'center');

% Reshaping into (time, y, x) for animation
U_dmd_reshaped = zeros(N_steps, n_grid_y, n_grid_x);
for tt = 1:N_steps
    snapshot = X_dmd_pred(:, tt);
    k=1;
    for j=1:n_grid_y
        for i=1:n_grid_x
            U_dmd_reshaped(tt,j,i) = snapshot(k);
            k = k+1;
        end
    end
end

%% Animation: DMD vs True Data

figure();
set(gcf, 'Position', [100 100 1100 500]);
sgtitle('DMD Reconstruction vs True DNS Data ($U_x$)', 'Interpreter', 'latex', 'FontSize', 14);
vmin = min(U_x_reshaped(:));
vmax = max(U_x_reshaped(:));

for tt = 1:num_timesteps
    subplot(2,1,1);
    contourf(X_grid, Y_grid, squeeze(U_x_reshaped(tt,:,:)), 40, 'LineStyle','none');
    colormap(gca,'jet');
    colorbar;
    caxis([vmin vmax]);
    title(sprintf('True Data (t = %.2f)', times_plot(tt)), 'Interpreter', 'latex');
    xlabel('X'); ylabel('Y');
    axis tight; set(gca,'YDir','normal');

    subplot(2,1,2);
    contourf(X_grid, Y_grid, squeeze(U_dmd_reshaped(tt,:,:)), 40, 'LineStyle','none');
    colormap(gca,'jet');
    colorbar;
    caxis([vmin vmax]);
    title(sprintf('DMD Reconstruction (t = %.2f)', times_plot(tt)), 'Interpreter', 'latex');
    xlabel('X'); ylabel('Y');
    axis tight; set(gca,'YDir','normal');

    drawnow;
end


%% Standard Operator Inference(Continuous Time)
r_inf = 50; 
U = U2(:,1:r_inf);
Phi = U; 
X_full = DataMatrix_Ux;
X_full_dot = (X_full(:, 2:end) - X_full(:, 1:end-1)) / delta_t;
X_full = X_full(:, 1:end-1);
X_full_dot = X_full_dot(:, 1:end);
U_r = Phi' * X_full; %  (r x N_timesteps-1)
% Reduced-order time derivatives
U_r_dot = Phi' * X_full_dot; %  (r x N_timesteps-1)
N_t = size(U_r, 2);
r = r_inf;
% Quadratic Term (Kronecker Product)
H_term = zeros(r*r, N_t);
for k = 1:N_t
u_r_k = U_r(:, k);
H_term(:, k) = kron(u_r_k, u_r_k); % (r^2 x 1)
end
% Z = [ U_r ; H_term ]
Z = [U_r; H_term]; % (r + r^2) x N_t
RHS = U_r_dot; %  r x N_t
%  G = [A_r | H_flat]
G = RHS * pinv(Z); %  r x (r + r^2)
% Extract the inferred operators
A_r = G(:, 1:r); % (r x r)
H_flat = G(:, r+1:end); %  (r x r^2)
% Reshaping the flat H_flat to 3D tensor H_r (r x r x r)
H_r = zeros(r, r, r);
for i = 1:r
H_r(i, :, :) = reshape(H_flat(i, :), [r, r]);
end
u_r0 = U_r(:, 1);
t_span = t_total;
ode_fun = @(t, u) OpInf_Model(u, A_r, H_r);
[~, U_r_pred_out] = ode45(ode_fun, t_span, u_r0); 
% Transpose  (r x N_timesteps)
U_r_pred = U_r_pred_out';
X_opinf = Phi * U_r_pred; % (N_grid x N_timesteps)
U_opinf_reshaped = zeros(num_timesteps, n_grid_y, n_grid_x);
for tt = 1:num_timesteps
snapshot = X_opinf(:, tt);
k=1;
for j=1:n_grid_y
for i=1:n_grid_x
U_opinf_reshaped(tt,j,i) = snapshot(k);
k = k+1;
end
end
end
%% Animation: OpInf Prediction vs True Data 
figure();
set(gcf, 'Position', [100 100 1100 500]);
sgtitle('OpInf Prediction vs True DNS Data ($U_x$)','Interpreter', 'latex', 'FontSize', 14);
v_min = min(U_x_reshaped(:));
v_max = max(U_x_reshaped(:));
for tt = 1:num_timesteps
subplot(2,1,1);
contourf(X_grid, Y_grid, squeeze(U_x_reshaped(tt,:,:)), 40, 'LineStyle', 'none');
colormap(gca, 'jet');
c = colorbar;
c.Label.String = '$U_x$ Velocity';
c.Label.Interpreter = 'latex';
title(sprintf('True Data (t = %.2f)', times_plot(tt)), 'Interpreter', 'latex');
xlabel('X'); ylabel('Y');
axis tight; set(gca, 'YDir', 'normal');
caxis([v_min v_max]);
subplot(2,1,2);
contourf(X_grid, Y_grid, squeeze(U_opinf_reshaped(tt,:,:)), 40, 'LineStyle', 'none');
colormap(gca, 'jet');
c = colorbar;
c.Label.String = '$U_x$ Velocity';
c.Label.Interpreter = 'latex';
title(sprintf('OpInf Prediction (t = %.2f)', times_plot(tt)), 'Interpreter', 'latex');
xlabel('X'); ylabel('Y');
axis tight; set(gca, 'YDir', 'normal');
caxis([v_min v_max]);
drawnow;
end

%% Energy-Scaled Residual-Corrected Discrete-Time Hybrid OpInf (ES-RCDTS-OpInf) 
X_total = DataMatrix_Ux; % (N_grid x N_timesteps)
U_r_T_actual_all=Phi'*X_total; 
Total_Model_func = @(t,u)OpInf_Model(u,A_r,H_r);
U_r_pred_T_ES = zeros(r, N_steps);
U_r_pred_T_ES(:,1) = U_r_T_actual_all(:,1);
U_r_T_actual_train = U_r_T_actual_all(:, 1:T_train);
mean_mode = mean(U_r_T_actual_all, 2); 
E_fluct_true_total=sum((U_r_T_actual_all-mean_mode).^2,1); 
E_pred_raw_fluct=zeros(1, N_steps);
E_pred_scaled_fluct = zeros(1, N_steps);

u_fluct_init = U_r_pred_T_ES(:,1) - mean_mode;
E_pred_raw_fluct(1) = sum(u_fluct_init.^2);
E_pred_scaled_fluct(1) = E_pred_raw_fluct(1);

%% Time-stepping with direct energy matching
for k = 1:N_steps-1
    u_current = U_r_pred_T_ES(:,k);  
    % Predicting Next Step (Unconstrained Integration)
    [~, u_step] = ode45(Total_Model_func, [t_span(k), t_span(k+1)], u_current);
    u_pred = u_step(end,:)';
    % Calculating Predicted Raw Energy (Fluctuation)
    u_fluct_pred_raw = u_pred - mean_mode;
    E_pred_fluct_raw = sum(u_fluct_pred_raw.^2);     
    E_actual_fluct_k1 = E_fluct_true_total(k+1);    
    % Computing Energy Scaling Factor
    scale_factor = sqrt(E_actual_fluct_k1 / (E_pred_fluct_raw + eps));
    scale_factor = max(min(scale_factor, 1.5), 0.5); 
    u_fluct_scaled = scale_factor * u_fluct_pred_raw;
    u_corrected = mean_mode + u_fluct_scaled; % Reconstruct total state
    U_r_pred_T_ES(:,k+1) = u_corrected;
    E_pred_raw_fluct(k+1) = E_pred_fluct_raw;
    E_pred_scaled_fluct(k+1) = sum(u_fluct_scaled.^2); 
end

X_reconstructed = Phi * U_r_pred_T_ES; 

%% Fluctuation Energy Comparison 
figure(); clf;
set(gcf,'Position',[100 100 800 500]);
hold on; grid on;
plot(t_span, E_fluct_true_total, 'k-', 'LineWidth', 2.5, 'DisplayName','Actual Fluctuation Energy (DNS)');
plot(t_span, E_pred_raw_fluct, 'r--', 'LineWidth', 1.5, 'DisplayName','Raw OpInf Fluctuation');
plot(t_span, E_pred_scaled_fluct, 'b-', 'LineWidth', 2, 'DisplayName','Energy-Scaled Fluctuation (Matched)');
plot([t_total(T_train) t_total(T_train)], [0 max(E_fluct_true_total)*1.1], 'r--', 'DisplayName', 'Training/Test Split');
xlabel('Time','Interpreter','latex');
ylabel('Reduced Fluctuation Energy $E'' = ||u_r''||_2^2$','Interpreter','latex');
title('Fluctuation Energy Evolution: Direct Energy Matching (ES-RCDTS-OpInf)','Interpreter','latex');
legend('show','Location','best');
hold off;

%% CALCULATE ALL REQUIRED FLUCTUATION FIELDS (CONSISTENT MEAN) 

% Get the Consistent Mean 

% U_r_T_actual_all is Phi' * X_total (full reduced data)
U_r_T_actual_train = U_r_T_actual_all(:, 1:T_train);
mean_mode_reduced = mean(U_r_T_actual_train, 2);
X_mean_full_DNS_consistent = Phi * mean_mode_reduced;
X_total = DataMatrix_Ux; % Full DNS data
X_fluct_actual = X_total - X_mean_full_DNS_consistent;
N_steps = size(X_total, 2);
U_fluct_actual_reshaped = zeros(N_steps, n_grid_y, n_grid_x);
for tt = 1:N_steps
snapshot = X_fluct_actual(:,tt);
k=1;
for j=1:n_grid_y
for i=1:n_grid_x
U_fluct_actual_reshaped(tt,j,i) = snapshot(k);
k = k+1;
end
end
end


% Calculate the Predicted Fluctuation Field (for U_fluct_ES_reshaped)
X_pred_T_ES = X_reconstructed;
X_pred_F_ES = X_pred_T_ES - X_mean_full_DNS_consistent;
U_fluct_ES_reshaped = zeros(N_steps, n_grid_y, n_grid_x);
for t_idx = 1:N_steps
snapshot = X_pred_F_ES(:,t_idx);
k = 1;
for j = 1:n_grid_y
for i = 1:n_grid_x
U_fluct_ES_reshaped(t_idx,j,i) = snapshot(k);
k = k + 1;
end
end
end

%% L2-Norm Error Calculation 
Error_L2_ES = zeros(1, N_steps);
X_true_total = X_total;
for tt = 1:N_steps
    error_vec = X_true_total(:, tt) - X_reconstructed(:, tt);
    numerator = norm(error_vec, 'fro'); 
    denominator = norm(X_true_total(:, tt), 'fro'); 
    Error_L2_ES(tt) = numerator / denominator;
end
figure(); clf; 
plot(t_total, Error_L2_ES, 'LineWidth', 2, 'DisplayName', 'ES-RCDTS-OpInf Error');
hold on;
plot([t_total(T_train) t_total(T_train)], [0 max(Error_L2_ES)*1.1], 'r--', 'DisplayName', 'Training/Test Split');
hold off;
title('ES-RCDTS-OpInf: Relative L2-Norm Error: Training vs. Test Performance', 'Interpreter', 'latex');
xlabel('Time ($t$)', 'Interpreter', 'latex');
ylabel('Relative Error $\frac{||U_{true} - U_{pred}||}{||U_{true}||}$', 'Interpreter', 'latex');
legend('show', 'Location', 'northwest');
grid on;
set(gca, 'YScale', 'log'); 
 T_test_start_index = T_train + floor(T_test/2);
 text(t_total(floor(T_train/2)), max(Error_L2_ES)*1.05, 'Training Region', 'Color', [0 0.5 0], 'FontSize', 10, 'HorizontalAlignment', 'center');
 text(t_total(T_test_start_index), max(Error_L2_ES)*1.05, 'Test/Forecast Region', 'Color', [0.8 0 0], 'FontSize', 10, 'HorizontalAlignment', 'center');

%% Animation: Actual vs Energy-Scaled OpInf Fluctuations 

figure(32); clf; 
set(gcf, 'Position', [100 100 1100 500]);
sgtitle('Fluctuation Field Comparison: Actual vs. Energy-Scaled OpInf', 'Interpreter', 'latex', 'FontSize', 14);
v_min_fluct = min([U_fluct_actual_reshaped(:); U_fluct_ES_reshaped(:)]);
v_max_fluct = max([U_fluct_actual_reshaped(:); U_fluct_ES_reshaped(:)]);
max_t_plot = N_steps; 

for tt = 1:max_t_plot
    subplot(2,1,1);
    contourf(X_grid, Y_grid, squeeze(U_fluct_actual_reshaped(tt,:,:)), 40, 'LineStyle','none');
    colormap(gca, 'jet'); c = colorbar; c.Label.String = '$U_x''$ (Fluctuation)'; c.Label.Interpreter = 'latex';
    caxis([v_min_fluct v_max_fluct]);
    title(sprintf('Actual Fluctuation (t = %.2f)', times_plot(tt)), 'Interpreter', 'latex');
    xlabel('X'); ylabel('Y'); axis tight; set(gca, 'YDir', 'normal');
    subplot(2,1,2);
    contourf(X_grid, Y_grid, squeeze(U_fluct_ES_reshaped(tt,:,:)), 40, 'LineStyle','none');
    colormap(gca, 'jet'); c = colorbar; c.Label.String = '$U_x''$ (Fluctuation)'; c.Label.Interpreter = 'latex';
    caxis([v_min_fluct v_max_fluct]);
    title(sprintf('ES-RCDTS-OpInf Fluctuation (t = %.2f, scaled)', times_plot(tt)), 'Interpreter', 'latex');
    xlabel('X'); ylabel('Y'); axis tight; set(gca, 'YDir', 'normal');
    drawnow;
end


%% L2 Error Comparison 
err_OpInf = zeros(num_timesteps,1); % Standard OpInf (Total Field Error)

for tt = 1:num_timesteps
    true_field = squeeze(U_x_reshaped(tt,:,:));
    opinf_field = squeeze(U_opinf_reshaped(tt,:,:));
    err_OpInf(tt) = norm(true_field(:) - opinf_field(:), 'fro') / norm(true_field(:), 'fro');
end

figure(); clf;
plot(times_plot(1:num_timesteps), err_OpInf, '-o', 'LineWidth', 1.6, 'DisplayName', 'Standard OpInf (Total Error)');
hold on;
plot(times_plot(1:num_timesteps), Error_L2_ES, '-s', 'LineWidth', 1.6, 'DisplayName', 'ES-RCDTS-OpInf (Fluctuation Error)');
plot(times_plot(1:num_timesteps), Error_L2_DMD, '-^', 'LineWidth', 1.6, 'DisplayName', 'DMD');
grid on;
xlabel('Time', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$L_2$ Error (normalized)', 'Interpreter', 'latex', 'FontSize', 12);
legend('Interpreter', 'latex', 'Location', 'best');
title('Error Evolution over Time', 'Interpreter', 'latex', 'FontSize', 14);


% =========================================================================
%                       HELPER FUNCTION DEFINITIONS
% =========================================================================
function dudt = OpInf_Model(u, A_r, H_r)
    % OpInf ROM: du/dt = A_r * u + H_r * (u \otimes u)
    r = length(u);
    L = A_r * u;
    
    % Quadratic term 
    Q = zeros(r, 1);
    u_tensor = u * u'; 
    for i = 1:r
        H_i = squeeze(H_r(i, :, :)); 
        Q(i) = sum(sum(H_i .* u_tensor));
    end
    dudt = L + Q;
end






%% Skin Friction Coefficient (Cf) Comparison Plot


U_inf_val = 1.0;        % Reference velocity scale
nu_val = 1.25e-3;       % Viscosity (Kinematic)
rho_val = 1.0;          % Density 
mu_val = rho_val * nu_val; % Dynamic Viscosity

y_min_val = Y_grid(1, 1);
y_1_val = Y_grid(2, 1);
dy_wall = y_1_val - y_min_val; 

x_values = X_grid(1, :); 
Rex = (U_inf_val * x_values) / nu_val; 
Cf_mean_actual = calculate_cf_local(U_x_reshaped, mu_val, rho_val, U_inf_val, dy_wall);
Cf_mean_dmd = calculate_cf_local(U_dmd_reshaped, mu_val, rho_val, U_inf_val, dy_wall);
Cf_mean_opinf = calculate_cf_local(U_opinf_reshaped, mu_val, rho_val, U_inf_val, dy_wall);
Cf_laminar = 0.664 ./ sqrt(Rex);
Cf_turbulent = 0.445 ./ (log(0.06 * Rex)).^2;

figure(); clf;
set(gcf,'Position',[100 100 1000 700]);

loglog(Rex, Cf_laminar, 'm--', 'DisplayName', 'Laminar (Blasius): $0.664/\sqrt{Re_x}$', 'LineWidth', 1.5);
hold on;
loglog(Rex, Cf_turbulent, 'c--', 'DisplayName', 'Turbulent (Schlichting): $0.445/(\log 0.06 Re_x)^2$', 'LineWidth', 1.5);

loglog(Rex, Cf_mean_actual, 'k.', 'DisplayName', 'DNS Data (Time-Avg)', 'MarkerSize', 10);
loglog(Rex, Cf_mean_dmd, 'b-', 'DisplayName', sprintf('DMD (r=%d)', r_dmd), 'LineWidth', 1.5);
loglog(Rex, Cf_mean_opinf, 'r-.', 'DisplayName', 'OpInf (Standard)', 'LineWidth', 1.5);

grid on;
xlabel('Local Reynolds Number ($Re_x$)', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Skin Friction Coefficient ($C_f$)', 'Interpreter', 'latex', 'FontSize', 14);
title('Skin Friction Coefficient: ROM Validation', 'Interpreter', 'latex', 'FontSize', 16);
legend('show', 'Location', 'best', 'Interpreter', 'latex');
set(gca, 'FontSize', 12);
axis tight;
hold off;


function Cf_mean = calculate_cf_local(U_field_reshaped, mu, rho, U_inf, dy_wall)
    U_y1 = squeeze(U_field_reshaped(:, 2, :)); 
    U_y0 = squeeze(U_field_reshaped(:, 1, :)); 
    dUx_dy_wall = (U_y1 - U_y0) / dy_wall; 
    Cf_array = (2 * mu / (rho * U_inf^2)) * dUx_dy_wall; 
    Cf_mean = mean(Cf_array, 1); 
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
