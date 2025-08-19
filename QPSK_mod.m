function S_mod = QPSK_mod(S)
    S = S(:)';
    S_mod = zeros( length(S) / 2, 1);
    bin_map = [0 0; 0 1; 1 1; 1 0];
    QPSK_constellation = [1+1i, -1+1i, -1-1i, 1-1i]/sqrt(2);

    for i = 1:length(S_mod)
        % Find the index in bin_map that matches the input bits
        [~, symbol_index] = ismember(S((i - 1) * 2 + 1:i * 2), bin_map, 'rows');
        % Map the index to the QPSK constellation
        S_mod(i) = QPSK_constellation(symbol_index);
    end
end