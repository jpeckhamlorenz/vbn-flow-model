function [t,x]  = VBN_flow_model_solver(ts, input_motor, input_beadwidth, IC, constants)

warning('off','all')

A_const = constants(1);  % forcing constant
B_const = constants(2);  % viscous friction constant
C_const = constants(3);  % pressure constant
D_const = constants(4);  % Coloumbic friction constant

N_const = constants(5);
M_const = constants(6);
U_const = constants(7);

N_INDEX = constants(8);
D_IN = constants(9);

a = constants(10);
b = constants(11);
c = constants(12);
d = constants(13);

% Regularization parameter for zero-velocity singularity.
% Smooths sign(v)*|v|^N_INDEX and sign(v) near v=0 to keep the Jacobian
% bounded, preventing ode15s Newton iteration stalls.
% Set to ~1% of typical steady-state |x(2)|. Tune if needed.
eps_reg = 1e-7;

t = [];
x = [];


%%
for i = 1:length(ts)-1

%    disp(i);
    if ts(i)<=0.1  % TYP: 0.2
        u1 = @(t) ((a.*log(exp(b.*(c-t))+1))./b)+t.*(a+d);
        u2 = @(t) input_beadwidth(i) + (input_beadwidth(i+1)-input_beadwidth(i))*(t-ts(i))/(ts(i+1)-ts(i));
    else
        u1 = @(t) input_motor(i) + (input_motor(i+1)-input_motor(i))*(t-ts(i))/(ts(i+1)-ts(i));
        u2 = @(t) input_beadwidth(i) + (input_beadwidth(i+1)-input_beadwidth(i))*(t-ts(i))/(ts(i+1)-ts(i));
        % u1 = @(t) interp1(ts(i:i+1),input_motor(i:i+1),t,'spline');
        % u2 = @(t) interp1(ts(i:i+1),input_beadwidth(i:i+1),t,'spline');
    end

    % x = [x(1),x(2)] = [theta,theta_dot]
    % x_dot = [theta_dot,theta_dot2]

    % x_dot = @(t, x) [...
    %     x(2);...
    %     A_const * (0 - x(1)) + ...
    %     1*B_const * U_const * x(2)^N_INDEX + ...
    %     1*C_const * (N_const * ((0^(-3*N_INDEX)-D_IN^(-3*N_INDEX))/(D_IN - 0)) * x(2)^N_INDEX + M_const * x(2)^N_INDEX) + ...
    %     1*D_const];

    % x_dot = @(t, x) [...
    %    x(2);...
    %    A_const * (u1(t) - x(1)) + ...
    %    1*B_const * U_const * x(2)^N_INDEX + ...
    %    0.5*C_const * (N_const * ((u2(t)^(-3*N_INDEX)-D_IN^(-3*N_INDEX))/(D_IN - u2(t))) * x(2)^N_INDEX + M_const * x(2)^N_INDEX) + ...
    %    1*D_const];

    % Regularized RHS: replaces sign(v)*|v|^n with v*(v^2+eps^2)^((n-1)/2)
    % and sign(v) with v/sqrt(v^2+eps^2). Identical to original for |v|>>eps_reg,
    % but Jacobian stays bounded near v=0.
    vel_pow = @(v) v .* (v.^2 + eps_reg^2).^((N_INDEX - 1) / 2);   % smooth sign(v)*|v|^N_INDEX
    vel_sgn = @(v) v ./ sqrt(v.^2 + eps_reg^2);                     % smooth sign(v)

    x_dot = @(t, x) [...
        x(2);...
        A_const * (u1(t) - x(1)) + ...
        1*B_const * U_const * vel_pow(x(2)) + ...
        1*C_const * (N_const * ((u2(t)^(-3*N_INDEX)-D_IN^(-3*N_INDEX))/(D_IN - u2(t))) * vel_pow(x(2)) + M_const * vel_pow(x(2))) + ...
        1*D_const * vel_sgn(x(2))];

    % x_dot = @(t, x) [...
    %     real(x(2));...
    %     real(A_const * (u1(t) - x(1))) + ...
    %     real(1*B_const * U_const * x(2)^N_INDEX) + ...
    %     real(1*C_const * (N_const * ((u2(t)^(-3*N_INDEX)-D_IN^(-3*N_INDEX))/(D_IN - u2(t))) * x(2)^N_INDEX + M_const * x(2)^N_INDEX)) + ...
    %     real(1*D_const)];

    tspan = [ts(i), ts(i+1)];

    [t_segment, x_segment] = ode15s(x_dot, tspan, IC);

    t = [t;t_segment]; %#ok<*AGROW>
    x = [x;x_segment];
    IC = x_segment(end,:);
end

end