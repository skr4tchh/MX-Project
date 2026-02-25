package kireiko.dev.millennium.ml;

import kireiko.dev.anticheat.checks.aim.ml.modules.RNN1Module;
import kireiko.dev.millennium.ml.data.module.ModuleML;
import kireiko.dev.millennium.ml.logic.Logger;
import kireiko.dev.millennium.ml.logic.Millennium;
import lombok.experimental.UtilityClass;

import java.util.Arrays;
import java.util.List;

@UtilityClass
public class ClientML {

    public static final boolean DEV_MODE = false;

    public static final String CLIENT_NAME = "quark-e-1.5-56k-public";
    private static final int TABLE_SIZE = 2;

    public static final List<ModuleML> MODEL_LIST = Arrays.asList(
                    new RNN1Module()
    );

    public void run() {
        long totalWeights = 0;
        for (int i = 0; i < MODEL_LIST.size(); i++) {
            final ModuleML moduleML = MODEL_LIST.get(i);
            Millennium loadedModel;

            if (DEV_MODE) {
                loadedModel = FactoryML.loadFromFile(i, moduleML.getName() + ".dat",
                                TABLE_SIZE, moduleML.getParameterBuffer(), moduleML.getVersion());
            } else {
                loadedModel = FactoryML.loadFromResources(i, moduleML.getName() + ".dat",
                                TABLE_SIZE, moduleML.getParameterBuffer(), moduleML.getVersion());
            }

            if (loadedModel != null) {
                totalWeights += loadedModel.parameters();
            }
        }
        Logger.info(CLIENT_NAME + " loaded!");
        Logger.info("Weights count: " + String.format("%, d", totalWeights));
    }
}
