%% ========== parameters ==========
clear; rng(0);
N        = 64;           % subcarriers
cp_len   = 16;           % CP length (>= L-1)
num_sym  = 100;          % number of OFDM symbols
M        = 4;            % QPSK
EbN0_dB  = 15;           % Eb/N0 (dB)

%% ========== channel (time domain) ==========
h = [0.9+0j, 0.5+0.3j, 0.2+0.1j];  % length L <= cp_len+1
L = length(h);
g = [h(:); zeros(N-length(h),1)]; % pad to length N (column vector)

% build single-symbol circulant matrix G (N x N)
G = zeros(N,N);
for i = 1:N
    G(:,i) = circshift(g, i-1);
end

% Big block-diagonal time-domain channel (N*num_sym x N*num_sym)
BigG = kron(speye(num_sym), G);  % sparse-friendly

%% ========== unitary DFT matrices ==========
% W = unnormalized DFT matrix; F_unitary = W / sqrt(N)
W = fft(eye(N));                 % W(k,n) = e^{-j2pi k n / N}
F = W / sqrt(N);                 % unitary DFT (F * F' = I)
BigF = kron(speye(num_sym), F);  % block-diag of F

% compute eigenvalues (per-subcarrier freq response)
H_diag = fft(g);                 % length N vector, H[k] = sum g_n e^{-j2pi k n / N}

%% ========== Tx: bits -> QPSK -> OFDM grid -> IFFT (time-domain) ==========
bits      = randi([0 1], num_sym*N*log2(M), 1);
sym_idx   = bi2de(reshape(bits,[],log2(M)), 'left-msb');
bb_sym    = pskmod(sym_idx, M, pi/4);
ofdm_grid = reshape(bb_sym, N, num_sym);   % frequency-domain symbols (N x S)

% Tx time-domain (per symbol)
tx_ifft = ifft(ofdm_grid, N, 1);    % N x num_sym  (each column is time-domain symbol, no CP)

%% ========== simulate time-domain channel using CP + linear conv per symbol ==========
% For each symbol: add CP (N+cp_len), linear conv with h (length L), then
% extract samples cp_len+1 : cp_len+N -> this yields circular convolution result.
rx_noCP_mat = zeros(N, num_sym);
for s = 1:num_sym
    tx_sym = tx_ifft(:, s);                     % N x 1
    tx_cp  = [ tx_sym(end-cp_len+1:end); tx_sym ];   % (N+cp_len) x 1
    y_full = conv(tx_cp, h.');                   % length (N+cp_len + L -1)
    % after receiving, remove first cp_len samples and take next N samples
    rx_sym = y_full(cp_len+1 : cp_len+N);        % N x 1
    rx_noCP_mat(:, s) = rx_sym;
end
% vectorized time-domain receive (stack columns)
rx_noCP_vec = rx_noCP_mat(:);   % (N*num_sym) x 1

%% ========== verify BigG * x_time_vec == rx_noCP_vec ==========
% Form X_time_vec (stacked transmitted time-domain symbols without CP)
X_time_mat = tx_ifft;           % N x num_sym
X_time_vec = X_time_mat(:);     % N*num_sym x 1

Y_viaBigG = BigG * X_time_vec;  % should equal rx_noCP_vec
err_time_conv = norm(Y_viaBigG - rx_noCP_vec);
fprintf('|| BigG*X_time - rx_noCP_vec || = %.3e\n', err_time_conv);

%% ========== compute frequency-domain via BigF and diagonalization ==========
% Frequency domain signals (stacked)
X_freq_vec = BigF * X_time_vec;       % (N*num_sym) x 1
Y_freq_vec_fromTime = BigF * rx_noCP_vec;

% Compute BigH = BigF * BigG * BigF'  (should be block-diagonal with diag(fft(g)))
BigH = BigF * BigG * BigF';
% Build explicit expected block-diag frequency matrix:
perBlock = diag(H_diag);   % N x N diagonal
ExpectedBigH = blkdiag( repmat({perBlock}, 1, num_sym) ); % dense; small S OK
% If num_sym is large, use kron: ExpectedBigH = kron(speye(num_sym), perBlock);

err_diag = norm(BigH - ExpectedBigH);
fprintf('|| BigH - blkdiag(diag(H)) || = %.3e\n', err_diag);

% Also check frequency-domain mapping: Y_freq_vec_fromTime vs BigH * X_freq_vec
Y_fromBigH = BigH * X_freq_vec;
err_freq_map = norm(Y_fromBigH - Y_freq_vec_fromTime);
fprintf('|| BigH*X_freq - FFT(rx_time) || = %.3e\n', err_freq_map);

%% ========== Add AWGN to time-domain received stream (optional) ==========
% Compute noise power consistent with constellation energy = 1
k = log2(M);
EsN0_dB = EbN0_dB + 10*log10(k);   % Es/N0
SNR = 10^(EsN0_dB/10);
% Assume Es=1 for QPSK: noise variance per complex sample:
sigma2 = 1/(2*SNR);
noise_vec = sqrt(sigma2)*(randn(size(rx_noCP_vec)) + 1j*randn(size(rx_noCP_vec)));
rx_noCP_vec_noisy = rx_noCP_vec + noise_vec;

% Frequency domain received (stacked)
Y_freq_noisy = BigF * rx_noCP_vec_noisy;
% Or compute per-symbol: RX_mat = fft(rx_noCP_mat + reshape(noise,...), N, 1)

%% ========== Equalization (1-tap per subcarrier) ==========
% Because BigH is block-diagonal with diag(H_diag), 1-tap eq is elementwise:
% X_hat_freq_vec = Y_freq_noisy ./ diag(BigH)  (blockwise)
% Extract diagonal of BigH:
BigH_diag = repmat(H_diag, num_sym, 1);  % length N*num_sym (stacked blocks)
% equalize:
X_hat_freq_vec = Y_freq_noisy ./ BigH_diag;

% Back to time domain:
X_hat_time_vec = BigF' * X_hat_freq_vec;  % inverse unitary DFT (BigF' is inverse)

% Recover symbols and bits
X_hat_mat = reshape(X_hat_freq_vec, N, num_sym);  % frequency domain per column
rx_sym_vec = X_hat_mat(:);
rx_idx = pskdemod(rx_sym_vec, M, pi/4);
rx_bits = de2bi(rx_idx, log2(M), 'left-msb');
rx_bits = rx_bits(:);

ber = mean(rx_bits ~= bits);
fprintf('BER (time-domain conv + BigG diag proof) = %.3e\n', ber);

%% ========== diagnostic plots and checks ==========
figure;
subplot(3,1,1);
plot(0:N-1, abs(H_diag)); title('|H[k]| (per-subcarrier magnitude)');
xlabel('k'); ylabel('|H[k]|'); grid on;

subplot(3,1,2);
imagesc(20*log10(abs(reshape(BigH_diag, N, num_sym))));
title('stacked H diag (N x num\_sym)'); colorbar;

subplot(3,1,3);
scatter(real(rx_sym_vec(1:1000)), imag(rx_sym_vec(1:1000)), 10, 'filled');
title('received symbols (first 1000)'); axis equal; grid on;

%% ========== summary of checks ==========
fprintf('\nSummary:\n - time-conv err = %.3e (should be ~0)\n - diagization err = %.3e (should be ~0)\n - freq mapping err = %.3e (should be ~0)\n - BER = %.3e\n\n', err_time_conv, err_diag, err_freq_map, ber);
