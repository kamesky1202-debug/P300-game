%% Neuro Dungeon P300 - Hackathon Entertainment Visualizer (single .m)
% -------------------------------------------------------------------------
% Story:
%   You move through a dungeon. Monsters emit signal flashes.
%   When the target stimulus appears and P300 evidence is strong, player attacks.
%
% Core analysis:
%   - 8ch CAR + 0.5-20Hz bandpass
%   - Epoching: -100ms..800ms (baseline correction)
%   - Auto channel: Pz(Ch5) vs Oz(Ch8), maximize Target-NonTarget separation
%
% Visual:
%   - Main scene: dungeon lane + monster encounter + attack effects
%   - Mini monitor (right-bottom): real waveform (Target/Non-target/Single-trial)
%   - Gauge and battle log

clear; clc; close all;
rng('shuffle');

%% ------------------------- Load -------------------------
[fn, fp] = uigetfile('*.mat', 'Select Unicorn P300 dataset .mat');
if isequal(fn,0), error('No file selected.'); end
S = load(fullfile(fp, fn));
if ~isfield(S,'y') || ~isfield(S,'trig')
    error('MAT file must contain y and trig.');
end
y = double(S.y);
trig = double(S.trig(:));
if isfield(S,'fs') && ~isempty(S.fs), fs = double(S.fs); else, fs = 256; end

if size(y,2) ~= 8, error('Expected 8 channels in y.'); end
if numel(trig) ~= size(y,1), error('trig length mismatch with y samples.'); end

%% ------------------------- Basic processing -------------------------
yCar = y - mean(y,2);
yF = bandpass(yCar, [0.5 20], fs);

[tCode, nCode] = inferCodes(trig);
tOn = find(trig == tCode);
nOn = find(trig == nCode);

pre = round(0.1*fs);
post = round(0.8*fs);
t = (-pre:post)'/fs;                % time vector
baseIdx = t < 0;
winIdx = (t >= 0.25) & (t <= 0.60); % P300 window

eT = cutEpochs(yF, tOn, pre, post); % [time x ch x trials]
eN = cutEpochs(yF, nOn, pre, post);
if isempty(eT) || isempty(eN), error('No valid epochs extracted.'); end

eT = baselineFix(eT, baseIdx);
eN = baselineFix(eN, baseIdx);

avgT = squeeze(mean(eT,3));
avgN = squeeze(mean(eN,3));

% Best channel between Pz(5) and Oz(8)
cand = [5 8];
score = zeros(1,numel(cand));
for i = 1:numel(cand)
    d = avgT(:,cand(i)) - avgN(:,cand(i));
    score(i) = max(d(winIdx));
end
[~,bi] = max(score);
bestCh = cand(bi);
bestName = ternary(bestCh==5,'Pz (Ch5)','Oz (Ch8)');

tarAvg = avgT(:,bestCh);
nonAvg = avgN(:,bestCh);
diffAvg = tarAvg - nonAvg;

noiseStd = std(diffAvg(baseIdx));
thr = max(1.5, 3.0*noiseStd);
[peakAmp,peakRel] = max(diffAvg(winIdx));
peakPos = find(winIdx);
peakIdx = peakPos(peakRel);
peakMs = t(peakIdx)*1000;

fprintf('\n=== Neuro Dungeon P300 ===\n');
fprintf('File: %s\n', fn);
fprintf('fs: %.1f Hz\n', fs);
fprintf('Target code: %g / Non-target code: %g\n', tCode, nCode);
fprintf('Selected channel: %s\n', bestName);
fprintf('P300 peak diff: %.2f uV at %.0f ms\n', peakAmp, peakMs);
fprintf('Threshold: %.2f uV\n', thr);

%% ------------------------- Game setup -------------------------
N_MONSTERS = 5;
MONSTER_HP = 140;
PLAYER_HP = 100;
maxSignalPerMonster = 6;
minSignalPerMonster = 4;

% Tempo tuning (larger = slower for presentation)
signalAnimFrames = 32;      % was 20
signalFramePause = 0.032;   % was 0.02
betweenSignalPause = 0.50;  % was 0.35
spawnPause = 0.90;          % was 0.6
betweenMonsterPause = 1.10; % was 0.8

% Subject-specific strength scores from your P300 ranking figure
% (used to make some heroes naturally stronger than others)
subjectPowerTable = struct( ...
    'S1', 0.04, ...
    'S2', 0.18, ...
    'S3', -0.02, ...
    'S4', 0.29, ...
    'S5', 1.20);

[~, stemName, ~] = fileparts(fn);
subjectTag = upper(stemName);
if isfield(subjectPowerTable, subjectTag)
    baseStrength = subjectPowerTable.(subjectTag);
else
    baseStrength = 0.15; % fallback for unknown subject names
end
sessionStrength = max(0, peakAmp - thr);
heroStrength = baseStrength + 0.55 * sessionStrength;
heroTier = classifyHeroTier(heroStrength);

fprintf('Subject tag: %s\n', subjectTag);
fprintf('Hero strength index: %.2f (%s)\n', heroStrength, heroTier);

% We build trial pools from real epochs
trialT = squeeze(eT(:,bestCh,:));   % [time x Nt]
trialN = squeeze(eN(:,bestCh,:));   % [time x Nn]
if isvector(trialT), trialT = trialT(:); end
if isvector(trialN), trialN = trialN(:); end
Nt = size(trialT,2);
Nn = size(trialN,2);

% Colors/theme
bg = [0.01 0.01 0.02];
dungeon = [0.08 0.1 0.12];
targetC = [0.0 1.0 1.0];
nonC = [0.25 0.25 0.30];
trialC = [1.0 0.85 0.1];
gridC = [0.1 0.4 0.15];
enemyC = [0.9 0.2 0.3];
heroC = [0.2 0.9 0.4];

%% ------------------------- Figure layout -------------------------
fig = figure('Color', bg, 'Name', 'NEURO DUNGEON // P300 QUEST', ...
    'Position', [40 60 1550 860]);

% Main encounter scene
axScene = axes('Parent', fig, 'Position', [0.04 0.08 0.62 0.84]);
hold(axScene,'on'); axis(axScene,[0 100 0 100]); axis(axScene,'off');
rectangle(axScene,'Position',[0 0 100 100],'FaceColor',dungeon,'EdgeColor','none');
for g = 10:10:90
    plot(axScene,[0 100],[g g],'-','Color',[0.07 0.12 0.1],'LineWidth',1);
end
for g = 10:10:90
    plot(axScene,[g g],[0 100],'-','Color',[0.07 0.12 0.1],'LineWidth',1);
end
text(axScene, 3, 96, 'NEURO DUNGEON', 'Color', [0.2 1 0.8], ...
    'FontSize', 22, 'FontWeight', 'bold', 'FontName', 'Consolas');

% Hero and dragon-style monster sprites
hero = createHeroSprite(axScene, 17, 46, 1.0);
dragon = createDragonSprite(axScene, 82, 47, 1.0);
heroGlow = patch(axScene, nan, nan, [0.2 1 0.5], ...
    'FaceAlpha', 0.08, 'EdgeColor', [0.6 1 0.7], 'LineStyle', ':', 'LineWidth', 1.2);
updateAura(heroGlow, hero.cx, hero.cy, 9);

% HP bars
rectangle(axScene, 'Position', [5 90 32 4], 'EdgeColor', [0.9 0.9 0.9], 'LineWidth', 1.2);
hPlayerBar = rectangle(axScene, 'Position', [5 90 32 4], 'FaceColor', [0.15 0.9 0.35], 'EdgeColor', 'none');
txtPlayer = text(axScene, 21, 92, 'PLAYER 100%', 'Color', [0.95 1 0.95], ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontName', 'Consolas');

rectangle(axScene, 'Position', [63 90 32 4], 'EdgeColor', [0.9 0.9 0.9], 'LineWidth', 1.2);
hEnemyBar = rectangle(axScene, 'Position', [63 90 32 4], 'FaceColor', [1 0.2 0.35], 'EdgeColor', 'none');
txtEnemy = text(axScene, 79, 92, 'MONSTER 100%', 'Color', [1 0.9 0.9], ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontName', 'Consolas');

% Log and banner
txtFloor = text(axScene, 3, 88, sprintf('Floor 1/%d', N_MONSTERS), 'Color', [0.9 1 0.9], ...
    'FontSize', 13, 'FontWeight', 'bold', 'FontName', 'Consolas');
txtLog = text(axScene, 3, 8, 'Dungeon started. Scanning monster signals...', ...
    'Color', [1 0.95 0.6], 'FontSize', 13, 'FontWeight', 'bold', 'FontName', 'Consolas');
txtAttack = text(axScene, 36, 70, '', 'Color', [1 0.3 0.2], 'FontSize', 30, ...
    'FontWeight', 'bold', 'FontName', 'Arial Black');

% Signal orbs from monster
orbHandles = gobjects(1,3);
for i = 1:3
    orbHandles(i) = plot(axScene, nan, nan, 'o', 'MarkerSize', 14, ...
        'MarkerFaceColor', [1 0.6 0.2], 'MarkerEdgeColor', [1 1 0.8], 'LineWidth', 1.3);
end

% Right-top stats panel
axStats = axes('Parent', fig, 'Position', [0.70 0.56 0.27 0.36], 'Color', bg);
axis(axStats,'off');
txtStatsTitle = text(axStats,0.02,0.95,'MISSION STATUS','Color',[0.2 1 0.8],...
    'FontSize',16,'FontWeight','bold','FontName','Consolas');
txtStats = text(axStats,0.02,0.83,'','Color',[0.85 1 0.9],...
    'FontSize',12,'FontName','Consolas','VerticalAlignment','top');

% Right-mid gauge
axGauge = axes('Parent', fig, 'Position', [0.72 0.43 0.22 0.10], 'Color', bg);
hold(axGauge,'on'); grid(axGauge,'on'); axGauge.GridColor = gridC;
axGauge.XColor = [0.8 1 0.8]; axGauge.YColor = [0.8 1 0.8];
ylim(axGauge,[0 100]); xlim(axGauge,[0.5 1.5]); xticks(axGauge,1); xticklabels(axGauge,{'Focus'});
hGauge = bar(axGauge, 1, 0, 0.55, 'FaceColor', [0.15 0.9 0.35], 'EdgeColor', [0.8 1 0.8]);
txtGauge = text(axGauge,1,5,'0%','HorizontalAlignment','center','Color',[0.9 1 0.9],...
    'FontWeight','bold','FontSize',12,'FontName','Consolas');
title(axGauge,'CONCENTRATION CHARGE','Color',[0.2 1 0.8],'FontWeight','bold');

% Right-bottom waveform monitor
axMon = axes('Parent', fig, 'Position', [0.69 0.08 0.29 0.30], 'Color', bg);
hold(axMon,'on'); grid(axMon,'on'); axMon.GridColor = gridC;
axMon.XColor = [0.8 1 0.8]; axMon.YColor = [0.8 1 0.8];
axMon.FontName = 'Consolas';
plot(axMon, t, nonAvg, '-', 'Color', nonC, 'LineWidth', 1.8);
plot(axMon, t, tarAvg, '-', 'Color', targetC, 'LineWidth', 2.0);
hTrial = plot(axMon, t, nan(size(t)), '-', 'Color', trialC, 'LineWidth', 1.8);
xline(axMon, 0, '--', 'Color', [0.7 0.7 0.7]);
xline(axMon, 0.25, ':', 'Color', [0.4 0.9 0.4]); xline(axMon,0.60,':','Color',[0.4 0.9 0.4]);
title(axMon, sprintf('Real Wave Monitor (%s)', bestName), 'Color', [0.2 1 0.8], 'FontWeight', 'bold');
xlabel(axMon, 'Time (s)', 'Color', [0.8 1 0.8]); ylabel(axMon, '\muV', 'Color', [0.8 1 0.8]);
legend(axMon, {'Non-target(avg)','Target(avg)','Current trial'}, ...
    'TextColor',[0.9 1 0.9],'Color',[0.05 0.1 0.05], 'EdgeColor',[0.3 0.6 0.3], 'FontSize',8);

%% ------------------------- Game loop -------------------------
kEnemy = 1;
cleared = 0;
finalEnemyHp = MONSTER_HP;
for m = 1:N_MONSTERS
    % Subject-adaptive enemy durability:
    % weak heroes get slightly lower enemy HP, strong heroes fight tougher bosses.
    enemyHp = max(85, MONSTER_HP * (0.92 + 0.18*min(heroStrength, 1.2)));
    txtFloor.String = sprintf('Floor %d/%d', m, N_MONSTERS);
    txtLog.String = sprintf('Monster %d appeared! Waiting for signal...', m);
    spawnY = 47 + 2*randn;
    updateDragonSprite(dragon, 82, spawnY, 1.0, 0);
    drawnow; pause(spawnPause);

    nSignals = randi([minSignalPerMonster maxSignalPerMonster]);
    targetSlot = randi([2 nSignals-1]); % avoid too early/late
    attacked = false;

    for s = 1:nSignals
        isTargetStim = (s == targetSlot);
        % Pick a real epoch waveform
        if isTargetStim
            idx = randi(Nt);
            w = trialT(:,idx);
            stimLabel = 'TARGET SIGNAL';
        else
            idx = randi(Nn);
            w = trialN(:,idx);
            stimLabel = 'NON-TARGET SIGNAL';
        end

        % Compute evidence from this single trial
        ev = max(w(winIdx)) - mean(w(baseIdx));
        gauge = min(100, max(0, 100 * ev / max(thr, eps)));

        % Monster sends signal animation
        txtLog.String = sprintf('Monster emits %s (%d/%d)', stimLabel, s, nSignals);
        for f = 1:signalAnimFrames
            x0 = 82 - f*1.3;
            y0 = 47 + 8*sin(0.35*f + 0.7*s);
            updateDragonSprite(dragon, 82, spawnY + 0.8*sin(0.4*f), 1.0, f);
            for oi = 1:3
                set(orbHandles(oi), 'XData', x0-(oi-1)*1.6, 'YData', y0+(oi-2)*2.8, ...
                    'MarkerSize', 18-3*oi, ...
                    'MarkerFaceColor', ternary(isTargetStim,[0 1 1],[1 0.6 0.2]));
            end

            % live waveform reveal on monitor
            kk = max(2, round(numel(t)*f/signalAnimFrames));
            set(hTrial, 'YData', [w(1:kk); nan(numel(t)-kk,1)]);

            % gauge update
            gNow = min(100, gauge * (f/signalAnimFrames));
            hGauge.YData = gNow;
            hGauge.FaceColor = [0.15+0.8*(gNow/100), 0.9*(gNow/100), 0.35];
            txtGauge.String = sprintf('%2.0f%%', gNow);
            txtGauge.Position(2) = min(gNow+4, 96);

            % stats
            txtStats.String = sprintf([ ...
                'File: %s\n' ...
                'fs: %.1f Hz\n' ...
                'Best channel: %s\n' ...
                'Hero tier: %s\n' ...
                'Hero strength: %.2f\n' ...
                'P300 threshold: %.2f uV\n' ...
                'Current evidence: %.2f uV\n' ...
                'Stimulus: %s\n' ...
                'Monster HP: %d%%\n' ...
                'Player HP: %d%%'], ...
                fn, fs, bestName, heroTier, heroStrength, thr, ev, stimLabel, round(enemyHp), round(PLAYER_HP));

            drawnow;
            pause(signalFramePause);
        end

        % Decision logic: attack only if target and enough evidence
        if isTargetStim && ev >= thr
            attacked = true;
            % Subject-adaptive damage:
            % weak heroes ~20-30, strong heroes ~45-75 on successful target lock.
            ratio = max(0, ev/max(thr,eps));
            dmg = 16 + 12*ratio + 30*heroStrength;
            if strcmp(heroTier, 'Legend')
                dmg = dmg + 8;
            end
            dmg = min(78, max(18, dmg));
            enemyHp = max(0, enemyHp - dmg);
            finalEnemyHp = enemyHp;
            hEnemyBar.Position(3) = 32*(enemyHp/100);
            txtEnemy.String = sprintf('MONSTER %d%%', round(enemyHp));

            txtAttack.String = 'NEURAL STRIKE!';
            for flash = 1:18
                txtAttack.Color = [1, 0.2+0.7*rand, 0.1+0.6*rand];
                updateAura(heroGlow, hero.cx, hero.cy, 9 + 1.2*randn);
                heroGlow.LineWidth = 1.0 + 2.5*rand;
                heroGlow.EdgeColor = [0.2+0.8*rand 1 0.4+0.6*rand];
                hero.head.MarkerFaceColor = [1 0.95 0.5];
                updateHeroSprite(hero, hero.cx, hero.cy + 0.4*sin(0.8*flash), 1.0, flash);
                drawnow; pause(0.018);
            end
            hero.head.MarkerFaceColor = [1.0 0.9 0.7];
            txtAttack.String = '';

            if enemyHp <= 0
                txtLog.String = sprintf('Monster %d defeated by P300 attack!', m);
                for ex = 1:24
                    sc = max(0.18, 1.0 - 0.03*ex);
                    updateDragonSprite(dragon, 82+randn*0.9, spawnY+randn*0.9, sc, ex);
                    set([dragon.body dragon.wingL dragon.wingR dragon.tail dragon.horn], ...
                        'FaceColor', [1, 0.2+0.8*rand, 0.2+0.8*rand]);
                    drawnow; pause(0.015);
                end
                cleared = cleared + 1;
                break;
            end
        else
            % Miss / no attack: player gets chip damage
            % Non-target is frequent, so miss penalty is reduced.
            if isTargetStim
                missDmg = max(4, 9 - 3*heroStrength);      % missed true target
            else
                missDmg = max(1, 3 - 1.2*heroStrength);    % frequent non-target chip
            end
            PLAYER_HP = max(0, PLAYER_HP - missDmg);
            finalEnemyHp = enemyHp;
            hPlayerBar.Position(3) = 32*(PLAYER_HP/100);
            txtPlayer.String = sprintf('PLAYER %d%%', round(PLAYER_HP));
            txtLog.String = sprintf('No lock. Player took chip damage. (%d%%)', round(PLAYER_HP));
            if PLAYER_HP <= 0
                txtLog.String = 'Player collapsed. Mission failed.';
                break;
            end
        end

        pause(betweenSignalPause);
    end

    if PLAYER_HP <= 0
        break;
    end
    if ~attacked
        txtLog.String = sprintf('Monster %d escaped. Move to next room...', m);
    end
    pause(betweenMonsterPause);
end

%% ------------------------- Ending -------------------------
if PLAYER_HP > finalEnemyHp
    txtAttack.String = sprintf('WIN  (YOU: %d%%  vs  DRAGON: %d%%)', round(PLAYER_HP), round(finalEnemyHp));
    txtAttack.Color = [0.2 1 0.8];
    txtLog.String = 'Final judgment: WIN (player HP is higher)';
elseif PLAYER_HP == finalEnemyHp
    txtAttack.String = sprintf('DRAW  (YOU: %d%%  vs  DRAGON: %d%%)', round(PLAYER_HP), round(finalEnemyHp));
    txtAttack.Color = [1.0 0.9 0.25];
    txtLog.String = 'Final judgment: DRAW (equal HP)';
else
    txtAttack.String = sprintf('LOSE  (YOU: %d%%  vs  DRAGON: %d%%)', round(PLAYER_HP), round(finalEnemyHp));
    txtAttack.Color = [1 0.25 0.25];
    txtLog.String = 'Final judgment: LOSE (dragon HP is higher)';
end

fprintf('\nRun finished: cleared %d/%d, player HP %.0f%%, final dragon HP %.0f%%\n', ...
    cleared, N_MONSTERS, PLAYER_HP, finalEnemyHp);

%% ------------------------- Local functions -------------------------
function [tCode, nCode] = inferCodes(trig)
    u = unique(trig(:)); u = u(~isnan(u));
    nz = u(u~=0);
    if any(nz==1) && any(nz==-1)
        tCode = 1; nCode = -1; return;
    end
    if numel(nz) < 2
        error('Could not infer target/non-target from trig non-zero codes: %s', mat2str(nz'));
    end
    tCode = max(nz);
    nCode = min(nz);
end

function ep = cutEpochs(sig, onsets, pre, post)
    n = size(sig,1);
    ch = size(sig,2);
    L = pre+post+1;
    ep = zeros(L, ch, 0);
    for i = 1:numel(onsets)
        s = onsets(i)-pre; e = onsets(i)+post;
        if s>=1 && e<=n
            ep(:,:,end+1) = sig(s:e,:); %#ok<AGROW>
        end
    end
end

function ep = baselineFix(ep, baseMask)
    for i = 1:size(ep,3)
        b = mean(ep(baseMask,:,i),1);
        ep(:,:,i) = ep(:,:,i)-b;
    end
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end

function tier = classifyHeroTier(strength)
    if strength >= 0.95
        tier = 'Legend';
    elseif strength >= 0.45
        tier = 'Elite';
    elseif strength >= 0.15
        tier = 'Rookie';
    else
        tier = 'Novice';
    end
end

function hero = createHeroSprite(ax, cx, cy, sc)
    % Simple warrior-like character using patches
    hero.body = patch(ax, nan, nan, [0.2 0.85 0.35], 'EdgeColor', [0.9 1 0.9], 'LineWidth', 1.4);
    hero.cape = patch(ax, nan, nan, [0.1 0.3 0.9], 'EdgeColor', [0.6 0.8 1], 'LineWidth', 1.0);
    hero.sword = patch(ax, nan, nan, [0.85 0.9 1.0], 'EdgeColor', [1 1 1], 'LineWidth', 1.0);
    hero.helm = patch(ax, nan, nan, [0.8 0.8 0.9], 'EdgeColor', [1 1 1], 'LineWidth', 1.0);
    hero.head = plot(ax, nan, nan, 'o', 'MarkerSize', 9, ...
        'MarkerFaceColor', [1.0 0.9 0.7], 'MarkerEdgeColor', [0.2 0.1 0.1], 'LineWidth', 1.1);
    hero.eye = plot(ax, nan, nan, '.', 'Color', [0.05 0.05 0.05], 'MarkerSize', 16);
    hero.cx = cx; hero.cy = cy;
    updateHeroSprite(hero, cx, cy, sc, 0);
end

function updateHeroSprite(hero, cx, cy, sc, phase)
    bob = 0.5 * sin(phase * 0.4);
    cy = cy + bob;
    hero.cx = cx; hero.cy = cy;
    % body
    xb = cx + sc*[-3 -2 2 3 2 -2];
    yb = cy + sc*[-6 3 3 -6 -8 -8];
    set(hero.body, 'XData', xb, 'YData', yb);
    % cape
    xc = cx + sc*[-2 -6 -5 -2];
    yc = cy + sc*[2 0 -8 -4];
    set(hero.cape, 'XData', xc, 'YData', yc);
    % sword
    xs = cx + sc*[3.5 7.8 8.2 3.9];
    ys = cy + sc*[1.2 4.5 3.9 0.6];
    set(hero.sword, 'XData', xs, 'YData', ys);
    % helm
    xh = cx + sc*[-1.8 0 1.8 1.2 -1.2];
    yh = cy + sc*[3.2 5.2 3.2 2.2 2.2];
    set(hero.helm, 'XData', xh, 'YData', yh);
    set(hero.head, 'XData', cx, 'YData', cy+1.8*sc);
    set(hero.eye, 'XData', cx+0.8*sc, 'YData', cy+1.8*sc);
end

function dragon = createDragonSprite(ax, cx, cy, sc)
    dragon.body = patch(ax, nan, nan, [0.82 0.25 0.3], 'EdgeColor', [1 0.75 0.75], 'LineWidth', 1.5);
    dragon.wingL = patch(ax, nan, nan, [0.55 0.15 0.2], 'EdgeColor', [0.95 0.6 0.65], 'LineWidth', 1.2);
    dragon.wingR = patch(ax, nan, nan, [0.55 0.15 0.2], 'EdgeColor', [0.95 0.6 0.65], 'LineWidth', 1.2);
    dragon.tail = patch(ax, nan, nan, [0.7 0.2 0.25], 'EdgeColor', [1 0.65 0.7], 'LineWidth', 1.1);
    dragon.horn = patch(ax, nan, nan, [0.95 0.9 0.6], 'EdgeColor', [1 1 0.8], 'LineWidth', 1.0);
    dragon.eye = plot(ax, nan, nan, 'o', 'MarkerSize', 7, ...
        'MarkerFaceColor', [1 1 0.2], 'MarkerEdgeColor', [0.1 0.1 0.1], 'LineWidth', 1.1);
    updateDragonSprite(dragon, cx, cy, sc, 0);
end

function updateDragonSprite(dragon, cx, cy, sc, phase)
    flap = 3.0 * sin(phase * 0.35);
    % body
    xb = cx + sc*[-8 -4 4 8 6 0 -6];
    yb = cy + sc*[-5 4 5 1 -4 -7 -6];
    set(dragon.body, 'XData', xb, 'YData', yb);
    % wings
    xwl = cx + sc*[-2 -11 -6 -1];
    ywl = cy + sc*[2 8+flap 0 0];
    set(dragon.wingL, 'XData', xwl, 'YData', ywl);
    xwr = cx + sc*[1 8 12 2];
    ywr = cy + sc*[2 0 6+flap 1];
    set(dragon.wingR, 'XData', xwr, 'YData', ywr);
    % tail
    xt = cx + sc*[-8 -13 -16 -12 -8];
    yt = cy + sc*[-3 -5 -2 1 -1];
    set(dragon.tail, 'XData', xt, 'YData', yt);
    % horn
    xh = cx + sc*[6 9 7];
    yh = cy + sc*[4 7 3];
    set(dragon.horn, 'XData', xh, 'YData', yh);
    set(dragon.eye, 'XData', cx+4.3*sc, 'YData', cy+2.4*sc);
end

function updateAura(hAura, cx, cy, r)
    th = linspace(0, 2*pi, 60);
    set(hAura, 'XData', cx + r*cos(th), 'YData', cy + r*sin(th));
end

