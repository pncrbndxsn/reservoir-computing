classdef OutputLayer < handle
    properties
        Wout  % 出力結合重み行列 [Ny, Nx]
    end

    methods
        % コンストラクタ
        function obj = OutputLayer(Nx, Ny)
            % Wout を標準正規分布に従う乱数で初期化
            obj.Wout = randn(Ny, Nx);
        end

        % call メソッド: モデル出力の計算
        function y = call(obj, x)
            y = obj.Wout * x;
        end

        % setWoutOpt メソッド: 学習後の出力結合重み行列を設定
        function setWoutOpt(obj, WoutOpt)
            obj.Wout = WoutOpt;
        end
    end
end
