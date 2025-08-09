classdef Feedback < handle
    properties
        Wfb  % フィードバック結合重み行列 [Nx, Ny]
    end

    methods
        % コンストラクタ
        function obj = Feedback(Nx, Ny, fbScaling)
            % Wfb を一様分布 [-fbScaling, fbScaling] に従う乱数で初期化
            obj.Wfb = (2*rand(Nx, Ny) - 1) * fbScaling;
        end

        % call メソッド: 出力層からリザバー層へのフィードバックを計算
        function Wfb_y = call(obj, y)
            Wfb_y = obj.Wfb * y;
        end
    end
end
