classdef InputLayer < handle
    properties
        Win  % 入力結合重み行列 [Nx, Nu]
    end

    methods
        % コンストラクタ
        function obj = InputLayer(Nu, Nx, inputScaling)
            % Win を一様分布 [-inputScaling, inputScaling] に従う乱数で初期化
            obj.Win = (2*rand(Nx, Nu) - 1) * inputScaling;
        end

        % call メソッド: 出力層からリザバー層への入力ベクトルを計算
        function Win_u = call(obj, u)
            Win_u = obj.Win * u;
        end
    end
end
