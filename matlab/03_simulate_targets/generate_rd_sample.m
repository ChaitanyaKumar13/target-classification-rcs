function RD = generate_rd_sample(className, Nr, Nd)
%GENERATE_RD_SAMPLE Create one synthetic RD map for a target class
% className: "drone" | "truck" | "tank"
% RD output: Nr x Nd

if nargin < 2, Nr = 256; end
if nargin < 3, Nd = 64; end

% base clutter/noise
RD = 0.6 * randn(Nr, Nd);

% pick a random target center (keep away from edges)
r0 = randi([30, Nr-30]);
d0 = randi([10, Nd-10]);

switch lower(string(className))
    case "drone"
        % small RCS but noticeable doppler spread (micro motion)
        amp = 10 + 3*rand();
        sig_r = 2 + rand();      % small in range
        sig_d = 4 + 2*rand();    % wider in doppler
        RD = RD + amp * gaussian2d(Nr, Nd, r0, d0, sig_r, sig_d);

        % micro-doppler sidebands (two weaker blobs)
        RD = RD + 0.5*amp * gaussian2d(Nr, Nd, r0, d0+6, sig_r, sig_d);
        RD = RD + 0.5*amp * gaussian2d(Nr, Nd, r0, d0-6, sig_r, sig_d);

    case "truck"
        % medium RCS and slightly spread in range (bigger object)
        amp = 16 + 4*rand();
        sig_r = 4 + rand();
        sig_d = 2 + rand();
        RD = RD + amp * gaussian2d(Nr, Nd, r0, d0, sig_r, sig_d);

    case "tank"
        % very high RCS, very concentrated, low doppler
        amp = 22 + 6*rand();
        sig_r = 2 + 0.5*rand();
        sig_d = 1 + 0.5*rand();
        d0 = randi([5, 15]); % low doppler bias
        RD = RD + amp * gaussian2d(Nr, Nd, r0, d0, sig_r, sig_d);

    otherwise
        error("Unknown className: %s", className);
end

% normalize like Carrada preprocessing style
RD = (RD - mean(RD(:))) / (std(RD(:)) + 1e-6);

end


% ---------- helper: 2D gaussian blob ----------
function G = gaussian2d(Nr, Nd, r0, d0, sig_r, sig_d)
[r, d] = ndgrid(1:Nr, 1:Nd);
G = exp(-(((r - r0).^2)/(2*sig_r^2) + ((d - d0).^2)/(2*sig_d^2)));
end
