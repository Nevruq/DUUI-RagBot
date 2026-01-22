package org.texttechnologylab.duui.examples;

import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;

import java.lang.reflect.Method;

public class SarcasmExample {

    public static void main(String[] args) throws Exception {
        int workers = 1;

        // DUUI context and composer
        DUUILuaContext ctx = new DUUILuaContext().withJsonLibrary();
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(ctx)
                .withWorkers(workers);

        // Docker-based sarcasm model (example image; replace with actual image you use)
        DUUIDockerDriver dockerDriver = new DUUIDockerDriver();
        composer.addDriver(dockerDriver);
        composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/german-sarcasm-bert:latest")
                .withScale(workers)
                .build());

        // Basic test text in German
        JCas jCas = JCasFactory.createText("Ich finde es wirklich toll, dass das jetzt passiert.", "de");
        // Create a single sentence for the whole text
        new Sentence(jCas, 0, jCas.getDocumentText().length()).addToIndexes();

        // Run the DUUI pipeline
        composer.run(jCas, "test");

        // Reflectively print any Sarcasm annotations if provided by the model
        try {
            Class<?> sarcasmClass = Class.forName("org.texttechnologylab.annotation.Sarcasm");
            @SuppressWarnings("unchecked")
            java.util.Collection<Object> list = (java.util.Collection<Object>) JCasUtil.select(jCas, (Class) sarcasmClass);
            if (list.isEmpty()) {
                System.out.println("No Sarcasm annotations found.");
            } else {
                for (Object s : list) {
                    for (Method m : sarcasmClass.getMethods()) {
                        if (m.getName().startsWith("get") && m.getParameterCount() == 0 && !m.getName().equals("getClass")) {
                            Object val = m.invoke(s);
                            System.out.println(m.getName().substring(3) + ": " + String.valueOf(val));
                        }
                    }
                }
            }
        } catch (ClassNotFoundException e) {
            System.out.println("Sarcasm annotation type not found in classpath.");
        }

        composer.shutdown();
    }
}