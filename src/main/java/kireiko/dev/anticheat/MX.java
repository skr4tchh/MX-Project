package kireiko.dev.anticheat;

import com.comphenix.protocol.ProtocolLibrary;
import com.comphenix.protocol.ProtocolManager;
import kireiko.dev.anticheat.api.data.PlayerContainer;
import kireiko.dev.anticheat.api.player.PlayerProfile;
import kireiko.dev.anticheat.commands.MXCommandHandler;
import kireiko.dev.anticheat.core.AsyncScheduler;
import kireiko.dev.anticheat.listeners.*;
import kireiko.dev.anticheat.managers.CheckManager;
import kireiko.dev.anticheat.services.AnimatedPunishService;
import kireiko.dev.anticheat.utils.ConfigCache;
import kireiko.dev.millennium.ml.ClientML;
import lombok.Getter;
import org.bukkit.Bukkit;
import org.bukkit.command.PluginCommand;
import org.bukkit.plugin.java.JavaPlugin;

import java.util.HashSet;
import java.util.Set;

public class MX extends JavaPlugin {

    public static final String
            command = "mx",
            name = "MX",
            permissionHead = "mx.",
            permission = permissionHead + "admin";

    @Getter
    private static MX instance;

    @Override
    public void onEnable() {
        instance = this;
        CheckManager.init();
        saveDefaultConfig();
        ConfigCache.loadConfig();
        kireiko.dev.anticheat.managers.DatasetManager.init();

        getLogger().info("Loading listeners...");
        loadListeners();
        getLogger().info("Booting timers...");
        punishTimer();
        getLogger().info("Initializing commands...");
        PluginCommand pCommand = this.getCommand(command);
        if (pCommand != null) {
            MXCommandHandler handler = new MXCommandHandler();
            pCommand.setExecutor(handler);
            pCommand.setTabCompleter(handler);
        }
        getLogger().info("Launching ML (Kireiko Millennium 5)...");
        ClientML.run();
        getLogger().info("Launched!");
    }

    private void punishTimer() {
        AnimatedPunishService.init();

        // reset vl
        Bukkit.getScheduler().runTaskTimerAsynchronously(this, () -> {
            float r = ConfigCache.VL_RESET;
            for (PlayerProfile profile : PlayerContainer.getUuidPlayerProfileMap().values()) {
                profile.fade(r);
                profile.setFlagCount(0);
            }
        }, 20L, 1200L);
    }

    private void loadListeners() {
        Bukkit.getPluginManager().registerEvents(new JoinQuitListener(), this);
        ProtocolManager protocolManager = ProtocolLibrary.getProtocolManager();
        protocolManager.addPacketListener(new RawMovementListener());
        protocolManager.addPacketListener(new UseEntityListener());
        protocolManager.addPacketListener(new EntityActionListener());
        protocolManager.addPacketListener(new VehicleTeleportListener());
    }

    @Override
    public void onDisable() {
        AsyncScheduler.shutdown();
    }

}
