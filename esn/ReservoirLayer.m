classdef ReservoirLayer < handle
    properties
        Wrec         % リザバー結合重み行列 [Nx, Nx]
        x            % リザバー状態ベクトル [Nx, 1]
        leakingRate  % リーク率
    end

    methods
        % コンストラクタ
        function obj = ReservoirLayer(Nx, networkDensity, spScaling, leakingRate)
            obj.x = (2*rand(Nx, 1) - 1) * 0.1;
            obj.leakingRate = leakingRate;

            % リザバー結合重み行列の計算
            mask = rand(Nx, Nx) < networkDensity;
            W = (2*rand(Nx, Nx) - 1) * 0.1 .* mask;
            spectralRadius = max(abs(eig(W)));  % スペクトル半径 (W の最大固有値の絶対値)
            obj.Wrec = W * (spScaling / spectralRadius);  % スケーリング
        end

        % call メソッド: リザバー状態ベクトルを更新
        function x = call(obj, Win_u)
            obj.x = (1 - obj.leakingRate) * obj.x + obj.leakingRate * tanh(obj.Wrec * obj.x + Win_u);
            x = obj.x;
        end
    end
end
