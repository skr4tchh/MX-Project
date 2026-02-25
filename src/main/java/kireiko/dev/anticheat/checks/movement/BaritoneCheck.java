package kireiko.dev.anticheat.checks.movement;

import kireiko.dev.anticheat.api.PacketCheckHandler;
import kireiko.dev.anticheat.api.data.ConfigLabel;
import kireiko.dev.anticheat.api.events.RotationEvent;
import kireiko.dev.anticheat.api.player.PlayerProfile;
import kireiko.dev.anticheat.managers.CheckManager;
import kireiko.dev.millennium.math.Euler;
import kireiko.dev.millennium.math.Simplification;
import kireiko.dev.millennium.math.Statistics;
import kireiko.dev.millennium.types.EvictingList;
import kireiko.dev.millennium.vectors.Vec2;
import kireiko.dev.millennium.vectors.Vec2f;
import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.*;

public class BaritoneCheck implements PacketCheckHandler {

    @Data
    @AllArgsConstructor
    private class BaritoneTermData {
        private double minH, maxH, minV, maxV, sum;
        private int constant;
    }

    private final PlayerProfile profile;
    private Map<String, Object> localCfg = new TreeMap<>();
    private final List<Vec2f> stack = new ArrayList<>();
    private final List<BaritoneTermData> longStack = new EvictingList<>(5);
    private int buffer = 0;
    private Vec2 oldMinMax = new Vec2(0, 0);

    @Override
    public ConfigLabel config() {
        localCfg.put("vl", 30);
        return new ConfigLabel("baritone", localCfg);
    }

    @Override
    public void applyConfig(Map<String, Object> params) {
        localCfg = params;
    }

    @Override
    public Map<String, Object> getConfig() {
        return localCfg;
    }

    public BaritoneCheck(PlayerProfile profile) {
        this.profile = profile;
        if (CheckManager.classCheck(this.getClass()))
            this.localCfg = CheckManager.getConfig(this.getClass());
    }

    @Override
    public void event(Object o) {
       if (o instanceof RotationEvent) {
           if (profile.isIgnoreFirstTick()) return;
           if (profile.getPlayer().getVehicle() != null) {
               return;
           }
           RotationEvent event = (RotationEvent) o;
           Vec2f delta = event.getDelta();
           if (Math.abs(delta.getY()) < 10 && Math.abs(profile.getTo().getPitch()) < 89.9f) {
               stack.add(delta);
               if (stack.size() >= 50) {
                   check();
               }
           }
       }
    }

    private void check() {
        { // logic
            float sum = 0;
            final List<Double> yStack = new ArrayList<>();
            final List<Double> xStack = new ArrayList<>();
            for (Vec2f f : stack) {
                sum += Math.abs(f.getX());
                yStack.add((double) f.getY());
                xStack.add((double) f.getX());
            }
            double minV = Statistics.getMin(yStack);
            double maxV = Statistics.getMax(yStack);
            double minH = Statistics.getMin(xStack);
            double maxH = Statistics.getMax(xStack);
            int distinct = Statistics.getDistinct(yStack);
            BaritoneTermData baritoneTermData = new BaritoneTermData(minH, maxH, minV, maxV, sum, distinct);
            longStack.add(baritoneTermData);
            { // result
                boolean strangePitch = isSame(minV, maxV) || (Math.abs(minV) < 0.009 && Math.abs(maxV) < 0.009) || (isSame(minV, oldMinMax.getX()) && isSame(maxV, oldMinMax.getY()));
                boolean invalidDistinct = distinct < 15 || (maxV < 0.009 && distinct < 50);
                boolean validSum = sum > 30 || (maxV < 0.009 && (sum > 4 || sum < 0.09));
                profile.debug("&8Baritone check result: [minV: " + Simplification.scaleVal(minV, 3)
                                + ", maxV: " + Simplification.scaleVal(maxV, 3)
                                + ", minH: " + Simplification.scaleVal(minH, 3)
                                + ", maxH: " + Simplification.scaleVal(maxH, 3)
                                + ", distinct: " + distinct
                                + ", sum: " + Simplification.scaleVal(sum, 3) + "]");
                if (strangePitch && invalidDistinct && validSum) {
                    buffer += 3;
                    if (buffer > 5) {
                        profile.punish("Movement", "Baritone", "Machine-like rotations" +
                                                        " [minV: " + Simplification.scaleVal(minV, 3)
                                                        + ", maxV: " + Simplification.scaleVal(maxV, 3)
                                                        + ", minH: " + Simplification.scaleVal(minH, 3)
                                                        + ", maxH: " + Simplification.scaleVal(maxH, 3)
                                                        + ", distinct: " + distinct
                                                        + ", sum: " + Simplification.scaleVal(sum, 3) + "]"
                                        ,
                                        ((Number) localCfg.get("vl")).floatValue() / 10.0f);
                        buffer = 3;
                    }
                } else if (buffer > 0) buffer--;
            }
            oldMinMax = new Vec2(minV, maxV);

        }
        stack.clear();
    }

    private boolean isSame(double a, double b) {
        return Math.abs(Math.abs(a) - Math.abs(b)) < 0.01;
    }

}
