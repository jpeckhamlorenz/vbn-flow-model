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

t = [];
x = [];


%%
for i = 1:length(ts)-1
    
    % disp(i);
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


    x_dot = @(t, x) [...
        x(2);...
        A_const * (u1(t) - x(1)) + ...
        1*sign(x(2))*B_const * U_const * abs(x(2))^N_INDEX + ...
        1*sign(x(2))*C_const * (N_const * ((u2(t)^(-3*N_INDEX)-D_IN^(-3*N_INDEX))/(D_IN - u2(t))) * abs(x(2))^N_INDEX + M_const * abs(x(2))^N_INDEX) + ...
        1*sign(x(2))*D_const];

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

