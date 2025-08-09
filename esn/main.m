%% Echo State Network (ESN) によるLorenz方程式の時系列データ予測プログラム

close all;
clear;
clc;

rng('default');

%% ESN モデルのパラメータ設定
Nu = 3;    % 入力次元
Nx = 200;  % リザバー層のノード数
Ny = 3;    % 出力次元

inputScaling = 0.1;    % 入力スケーリング
networkDensity = 0.1;  % リザバー層のネットワーク密度
spScaling = 0.95;      % Wrecのスペクトル半径のスケーリング
leakingRate = 1.0;     % リーク率
beta = 1e-4;           % リッジ回帰の正則化係数
fbScaling = [];        % フィードバックスケーリング
noiseLevel = [];       % ノイズレベル

delta = 1e-4;    % RLS の初期値パラメータ
lambda = 0.995;  % RLS の忘却係数
nUpdates = 5;    % RLS の更新回数

isBatch = true;  % バッチ学習かどうか

%% Lorenz方程式の時系列データの生成
T = 100;                % 全体時間
dt = 0.001;             % 時間刻み幅
x0 = [1.0; 1.0; 1.0];   % Lorenz方程式の初期値 [x; y; z]
param = [10, 28, 8/3];  % Lorenz方程式のパラメータ [sigma, r, b]

L = Lorenz(T, dt, x0, param);   % Lorenzオブジェクトの生成
learningData = L.RungeKutta();  % Runge-Kutta法で時系列データを生成

%% 学習データの設定
lenTrain = floor(length(learningData)*0.9);  % 訓練データの長さ
lenTest = length(learningData) - lenTrain;   % テストデータの長さ
lenTrans = 0;                                % 過渡期間

idxTestStart = lenTrain + 1;              % テスト開始時刻 [pts]
idxTestEnd = idxTestStart + lenTest - 1;  % テスト終了時刻 [pts]
idxTrainStart = idxTestStart - lenTrain;  % 学習開始時刻 [pts]
idxTrainEnd = idxTestStart - 1;           % 学習終了時刻 [pts]

tTrain = idxTrainStart:idxTrainEnd-1;     % 訓練データの時刻ベクトル
tTest  = idxTestStart+1:idxTestEnd;       % テストデータの時刻ベクトル

%% 学習データの作成
UTrain = learningData(idxTrainStart:idxTrainEnd-1, :);
DTrain = learningData(idxTrainStart+1:idxTrainEnd, :);

UTest = learningData(idxTestStart:idxTestEnd-1, :);
DTest = learningData(idxTestStart+1:idxTestEnd, :);

%% 標準化
[UTrain, muU, sigmaU] = normalize(UTrain, 1);
[DTrain, muD, sigmaD] = normalize(DTrain, 1);

UTest = (UTest - muU) ./ sigmaU;

%% ESNモデルの作成
model = ESN(Nu, Nx, Ny, inputScaling, networkDensity, spScaling, leakingRate, fbScaling, noiseLevel);

if isBatch
    % バッチ学習
    optimizer = Tikhonov(Nx, Ny, beta);
    YTrain = model.train(UTrain, DTrain, optimizer, lenTrans);

    % テストデータで予測
    Ypred = model.run(UTest) .* sigmaD + muD;

    % RMSE計算
    rmsePred = rmse(Ypred, DTest, 2);
else
    % オンライン学習
    U = [UTrain; UTest];
    D = [DTrain; DTest];
    optimizer = RLS(Nx, Ny, delta, lambda, nUpdates);
    [Y, WoutMeanAbs] = model.adapt(U, D, optimizer);

    % RMSE計算
    rmsePred = rmse(Y, D, 2);
end

%% プロット
t = tiledlayout(4,1);

% x(t)
nexttile; hold on;
plot(tTest, DTest(:,1), '-', 'LineWidth', 2.0);
plot(tTest, Ypred(:,1), '--', 'LineWidth', 2.0);
xlim([tTest(1), tTest(end)]);
ylabel('$x(t)$', Interpreter='latex');
legend('Target', 'Predict', Interpreter='latex');
set(gca, TickLabelInterpreter='latex', FontSize=16);
grid on;

% y(t)
nexttile; hold on;
plot(tTest, DTest(:,2), '-', 'LineWidth', 2.0);
plot(tTest, Ypred(:,2), '--', 'LineWidth', 2.0);
xlim([tTest(1), tTest(end)]);
ylabel('$y(t)$', Interpreter='latex');
legend('Target', 'Predict', Interpreter='latex');
set(gca, TickLabelInterpreter='latex', FontSize=16);
grid on;

% z(t)
nexttile; hold on;
plot(tTest, DTest(:,3), '-', 'LineWidth', 2.0);
plot(tTest, Ypred(:,3), '--', 'LineWidth', 2.0);
xlim([tTest(1), tTest(end)]);
ylabel('$z(t)$', Interpreter='latex');
legend('Target', 'Predict', Interpreter='latex');
set(gca, TickLabelInterpreter='latex', FontSize=16);
grid on;

% RMSE
nexttile; hold on;
plot(tTest, rmsePred, '-', 'LineWidth', 2.0);
xlim([tTest(1), tTest(end)]);
ylabel('RMSE', Interpreter='latex');
xlabel('Time Step [pts]', Interpreter='latex');
set(gca, TickLabelInterpreter='latex', FontSize=16);
grid on;
