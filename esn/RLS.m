classdef RLS < handle
    properties
        delta     % 共分散行列 P の初期値パラメータ
        lambda    % 忘却係数 (0 < lambda < 1)
        nUpdates  % 各時刻での更新回数
        P         % 共分散行列の逆数 [Nx, Nx]
        Wout      % 出力結合重み行列 [Ny, Nx]
    end

    methods
        % コンストラクタ
        function obj = RLS(Nx, Ny, delta, lambda, nUpdates)
            obj.delta = delta;
            obj.lambda = lambda;
            obj.nUpdates = nUpdates;
            obj.P = eye(Nx) ./ delta;
            obj.Wout = zeros(Ny, Nx);
        end

        % 出力結合重み行列 Wout の逐次更新 (RLSアルゴリズム)
        function Wout = call(obj, x, d)
            for i = 1:obj.nUpdates
                error = d - obj.Wout * x;  % 出力誤差

                % ゲインベクトルの計算
                Px = obj.P * x;
                gain = (1 / obj.lambda) * Px / (1 + (1 / obj.lambda) * (x' * Px));

                % 共分散行列 P の更新
                obj.P = (1 / obj.lambda) * (obj.P - gain * x' * obj.P);

                % Wout の更新
                obj.Wout = obj.Wout + error * gain';
            end

            Wout = obj.Wout;
        end
    end
end
