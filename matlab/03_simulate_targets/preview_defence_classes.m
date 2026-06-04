clc; clear;

Nr = 256; Nd = 64;

classes = ["drone","truck","tank"];

for i = 1:numel(classes)
    RD = generate_rd_sample(classes(i), Nr, Nd);

    figure;
    imagesc(RD);
    colormap jet;
    colorbar;
    xlabel("Doppler bins");
    ylabel("Range bins");
    title("Synthetic RD: " + classes(i));
end

disp("STEP 9 PREVIEW COMPLETE ✅ (3 defence classes visualized)");
