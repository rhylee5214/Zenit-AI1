// ZENIT desktop launcher. Starts the local ZENIT server if it is not running,
// waits for the model to load, then opens ZENIT in its own app window.
// Build with scripts\make_app.ps1 (uses the C# compiler that ships with Windows).
using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Threading;

class ZenitLauncher
{
    // The launcher talks to the server on 127.0.0.1; the app window opens on
    // http://localhost:8055 because Google sign-in only accepts origins
    // registered in Google Cloud, and http://localhost is the one that works.
    const string Url = "http://127.0.0.1:8055";
    const string AppUrl = "http://localhost:8055";

    static string Get(string path)
    {
        try
        {
            var req = (HttpWebRequest)WebRequest.Create(Url + path);
            req.Timeout = 3000;
            using (var resp = req.GetResponse())
            using (var r = new StreamReader(resp.GetResponseStream()))
                return r.ReadToEnd();
        }
        catch { return null; }
    }

    static int Main()
    {
        Console.Title = "ZENIT";
        string root = AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\');
        string python = Path.Combine(root, @".venv\Scripts\python.exe");

        if (Get("/v1/health") == null)
        {
            if (!File.Exists(python))
            {
                Console.WriteLine("Could not find " + python);
                Console.WriteLine("Keep ZENIT.exe in the ZENIT folder (next to the .venv folder).");
                Console.ReadKey();
                return 1;
            }
            Console.WriteLine("Starting the ZENIT server (its window opens minimized; close it to stop ZENIT)...");
            var psi = new ProcessStartInfo("cmd.exe",
                "/c title ZENIT server && \"" + python + "\" -m zenit.api --local --root \"" + root + "\"")
            {
                WorkingDirectory = root,
                UseShellExecute = true,
                WindowStyle = ProcessWindowStyle.Minimized,
            };
            Process.Start(psi);
        }
        else
        {
            Console.WriteLine("ZENIT server is already running.");
        }

        Console.Write("Loading the model (first start takes 1-2 minutes)");
        for (int i = 0; i < 300; i++)
        {
            string models = Get("/v1/models");
            if (models != null && models.Contains("\"default\":true") &&
                models.Contains("\"loaded\":true"))
                break;
            Console.Write(".");
            Thread.Sleep(2000);
        }
        Console.WriteLine();

        try { Process.Start(new ProcessStartInfo("msedge", "--app=" + AppUrl) { UseShellExecute = true }); }
        catch
        {
            try { Process.Start(new ProcessStartInfo("chrome", "--app=" + AppUrl) { UseShellExecute = true }); }
            catch { Process.Start(new ProcessStartInfo(AppUrl) { UseShellExecute = true }); }
        }
        return 0;
    }
}
