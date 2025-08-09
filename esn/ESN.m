classdef ESN < handle
    properties
        Input      % 入力層
        Reservoir  % リザバー層
        Output     % 出力層
        Nu         % 入力次元
        Nx         % リザバー層のノード数
        Ny         % 出力次元
        yprev      % 直前の出力
        Feedback   % フィードバック層
        noise      % ノイズ
    end

    methods
        % コンストラクタ
        function obj = ESN(Nu, Nx, Ny, inputScaling, networkDensity, rho, leakRate, fbScaling, noiseLevel)
            obj.Input = InputLayer(Nu, Nx, inputScaling);
            obj.Reservoir = ReservoirLayer(Nx, networkDensity, rho, leakRate);
            obj.Output = OutputLayer(Nx, Ny);
            obj.Nu = Nu;
            obj.Nx = Nx;
            obj.Ny = Ny;
            obj.yprev = zeros(Ny, 1);
            if ~isempty(fbScaling)
                obj.Feedback = Feedback(Nx, Ny, fbScaling);
            else
                obj.Feedback = [];
            end
            if ~isempty(noiseLevel)
                obj.noise = (2*rand(obj.Nx,1) - 1) .* noiseLevel;
            else
                obj.noise = [];
            end
        end

        % 訓練
        function Y = train(obj, UTrain, DTrain, optimizer, lenTrans)
            lenTrain = length(UTrain);
            Y = zeros(lenTrain, obj.Ny);

            for i = 1:lenTrain
                u = UTrain(i,:)';
                Win_u = obj.Input.call(u);

                % フィードバック
                if ~isempty(obj.Feedback)
                    Wfb_y = obj.Feedback.call(obj.yprev);
                    Win_u = Win_u + Wfb_y;
                end

                % ノイズ
                if ~isempty(obj.noise)
                    Win_u = Win_u + obj.noise;
                end
                % リザバー状態ベクトルの更新
                x = obj.Reservoir.call(Win_u);

                d = DTrain(i,:)';  % 教師データ

                % 過渡期間終了後に学習
                if i > lenTrans
                    optimizer.call(x, d);
                end

                % モデル出力の計算
                y = obj.Output.call(x);
                Y(i,:) = y';
                obj.yprev = d;
            end
            % 学習後の出力結合重み行列を設定
            obj.Output.setWoutOpt(optimizer.getWoutOpt());
        end

        % 予測
        function Ypred = predict(obj, UTest)
            lenTest = length(UTest);
            Ypred = zeros(lenTest, obj.Ny);

            for i = 1:lenTest
                u = UTest(i,:)';
                Win_u = obj.Input.call(u);

                % フィードバック
                if ~isempty(obj.Feedback)
                    Wfb_y = obj.Feedback.call(obj.yprev);
                    Win_u = Win_u + Wfb_y;
                end

                % リザバー状態ベクトルの更新
                x = obj.Reservoir.call(Win_u);

                % モデル出力の計算
                ypred = obj.Output.call(x);
                Ypred(i,:) = ypred';
                obj.yprev = ypred;
            end
        end

        % 自律予測
        function Yrun = run(obj, UTest)
            lenTest = length(UTest);
            Yrun = zeros(lenTest, obj.Ny);

            u = UTest(1,:)';
            for i = 1:lenTest
                Win_u = obj.Input.call(u);

                % フィードバック
                if ~isempty(obj.Feedback)
                    Wfb_y = obj.Feedback.call(obj.yprev);
                    Win_u = Win_u + Wfb_y;
                end

                % リザバー状態ベクトルの更新
                x = obj.Reservoir.call(Win_u);

                % モデル出力の計算
                yrun = obj.Output.call(x);
                Yrun(i,:) = yrun';
                obj.yprev = yrun;
                u = yrun;
            end
        end

        % オンライン学習 (RLS)
        function [Y, WoutMeanAbs] = adapt(obj, U, D, optimizer)
            lenData = length(U);
            Y = zeros(lenData, obj.Ny);

            WoutMeanAbs = zeros(lenData, 1);  % Woutの平均絶対値
            for i = 1:lenData
                u = U(i,:)';
                Win_u = obj.Input.call(u);

                % リザバー状態ベクトルの更新
                x = obj.Reservoir.call(Win_u);

                d = D(i,:)';  % 教師データ

                % 出力結合重み行列の更新
                Wout = optimizer.call(x, d);

                % モデル出力の計算
                y = Wout * x;
                Y(i,:) = y';
                WoutMeanAbs(i) = mean(abs(Wout(:)));
            end
        end
    end
end
