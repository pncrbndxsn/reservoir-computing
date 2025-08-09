classdef Tikhonov < handle
    properties
        Nx    % リザバー層のノード数
        X_XT  % x * x' の集積行列 [Nx, Nx]
        D_XT  % d * x' の集積行列 [Ny, Nx]
        beta  % リッジ回帰の正則化係数
    end

    methods
        % コンストラクタ
        function obj = Tikhonov(Nx, Ny, beta)
            obj.Nx = Nx;
            obj.X_XT = zeros(Nx, Nx);
            obj.D_XT = zeros(Ny, Nx);
            obj.beta = beta;
        end

        % 訓練データの累積和
        function obj = call(obj, x, d)
            obj.X_XT = obj.X_XT + (x * x');
            obj.D_XT = obj.D_XT + (d * x');
        end

        % 出力結合重み行列 Wout の計算
        function WoutOpt = getWoutOpt(obj)
            XPseudoInv = pinv(obj.X_XT + obj.beta * eye(obj.Nx));
            WoutOpt = obj.D_XT * XPseudoInv;
        end
    end
end
