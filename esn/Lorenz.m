classdef Lorenz
    properties
        T      % 総時間
        dt     % 時間刻み
        x0     % 初期値ベクトル
        sigma
        r
        b
    end

    methods
        function obj = Lorenz(T, dt, x0, param)
            obj.T = T;
            obj.dt = dt;
            obj.x0 = x0;  % 初期値を列ベクトルで格納
            obj.sigma = param(1);
            obj.r = param(2);
            obj.b = param(3);
        end

        function dXdt = LorenzEq(obj, ~, x)
            dxdt = obj.sigma * (x(2) - x(1));
            dydt = x(1) * (obj.r - x(3)) - x(2);
            dzdt = x(1) * x(2) - obj.b * x(3);
            dXdt = [dxdt; dydt; dzdt];
        end

        function X = RungeKutta(obj)
            N = floor(obj.T / obj.dt);
            X = zeros(N, 3);

            x = obj.x0;
            X(1, :) = x';
            for i = 2:N
                t = i * obj.dt;
                k1 = obj.LorenzEq(t, x);
                k2 = obj.LorenzEq(t + obj.dt/2, x + obj.dt/2 * k1);
                k3 = obj.LorenzEq(t + obj.dt/2, x + obj.dt/2 * k2);
                k4 = obj.LorenzEq(t + obj.dt, x + obj.dt * k3);
                x = x + obj.dt/6 * (k1 + 2*k2 + 2*k3 + k4);
                X(i, :) = x';
            end
        end
    end
end
