package org.broadinstitute.hellbender.tools.walkers.annotator;

import htsjdk.samtools.seekablestream.SeekablePathStream;
import htsjdk.tribble.readers.LineIterator;
import htsjdk.tribble.readers.PositionalBufferedStream;
import htsjdk.variant.variantcontext.VariantContext;
import htsjdk.variant.vcf.VCFCodec;
import htsjdk.variant.vcf.VCFHeader;
import org.broadinstitute.hellbender.CommandLineProgramTest;
import org.broadinstitute.hellbender.utils.Utils;
import org.broadinstitute.hellbender.utils.runtime.ProcessController;
import org.broadinstitute.hellbender.utils.runtime.ProcessOutput;
import org.broadinstitute.hellbender.utils.runtime.ProcessSettings;
import org.broadinstitute.hellbender.utils.test.ArgumentsBuilder;
import org.broadinstitute.hellbender.utils.test.IntegrationTestSpec;
import org.broadinstitute.hellbender.utils.test.VariantContextTestUtils;
import org.seqdoop.hadoop_bam.util.VCFHeaderReader;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;

public class VariantAnnotatorIntegrationTest extends CommandLineProgramTest {
    final static String STANDARD_ANNOTATIONS = " -G StandardAnnotation "; //-G StandardUG ?
    private static final List<String> ATTRIBUTES_TO_IGNORE = Arrays.asList(
            "QD",//TODO QD has a cap value and anything that reaches that is randomized.  It's difficult to reproduce the same random numbers across gatk3 -> 4
            "FS");//TODO There's some bug in either gatk3 or gatk4 fisherstrand that's making them not agree still, I'm not sure which is correct

    //TODO because of differences between how GATK3 and GATK4 handle capturing reads for spanning deletions (Namely 3 only looks for reads overlapping the first site, 4 gets all reads over the span)
    //TODO then we want ot ignore affected attributes for concordance tests
    private static final List<String> DEPTH_ATTRIBUTES_TO_IGNORE = Arrays.asList(
            "QD",
            "DP",
            "MQ");

    final String testFile = getToolTestDataDir() + "vcfexample2.vcf";

    public static String baseTestString() {
        return "-R " + b37_reference_20_21 + " -O %s ";
    }

    //==================================================================================================================
    // Testing
    //==================================================================================================================

    private void assertVariantContextsMatch(File input, File expected, List<String> extraArgs, String reference) throws IOException {
        final VCFHeader header = getHeaderFromFile(expected);

        runVariantAnnotatorAndAssertSomething(input, expected, extraArgs, (a, e) -> {
            VariantContextTestUtils.assertVariantContextsAreEqualAlleleOrderIndependent(a, e, ATTRIBUTES_TO_IGNORE, header);
        }, reference);
    }

    private void runVariantAnnotatorAndAssertSomething(File input, File expected, List<String> additionalArguments, BiConsumer<VariantContext, VariantContext> assertion, String reference) throws IOException {
        final File output = createTempFile("variantAnnotator", ".vcf");

        final ArgumentsBuilder args = new ArgumentsBuilder();
        args.addReference(new File(reference))
                .addOutput(output);
        args.addArgument("V", input.getAbsolutePath());


        // Handling a difference in syntax between GATK3 and GATK4 wrt. annotation groups
        additionalArguments = additionalArguments.stream().map(a -> a.contains("Standard") ? a + "Annotation" : a).collect(Collectors.toList());
        additionalArguments.forEach(args::add);

        Utils.resetRandomGenerator();
        runCommandLine(args);

        final List<VariantContext> expectedVC = getVariantContexts(expected);
        final List<VariantContext> actualVC = getVariantContexts(output);
        assertForEachElementInLists(actualVC, expectedVC, assertion);
    }

    /**
     * Returns a list of VariantContext records from a VCF file
     *
     * @param vcfFile VCF file
     * @return list of VariantContext records
     * @throws IOException if the file does not exist or can not be opened
     */
    private static List<VariantContext> getVariantContexts(final File vcfFile) throws IOException {
        final VCFCodec codec = new VCFCodec();
        final FileInputStream s = new FileInputStream(vcfFile);
        final LineIterator lineIteratorVCF = codec.makeSourceFromStream(new PositionalBufferedStream(s));
        codec.readHeader(lineIteratorVCF);

        final List<VariantContext> VCs = new ArrayList<>();
        while (lineIteratorVCF.hasNext()) {
            final String line = lineIteratorVCF.next();
            Assert.assertFalse(line == null);
            VCs.add(codec.decode(line));
        }

        return VCs;
    }

    private static VCFHeader getHeaderFromFile(final File vcfFile) throws IOException {
        SeekablePathStream stream = new SeekablePathStream(vcfFile.toPath());
        VCFHeader header = VCFHeaderReader.readHeaderFrom(new SeekablePathStream(vcfFile.toPath()));
        stream.close();
        return header;
    }

    private static <T> void assertForEachElementInLists(final List<T> actual, final List<T> expected, final BiConsumer<T, T> assertion) {
        Assert.assertEquals(actual.size(), expected.size(), "different number of elements in lists:\n"
                + actual.stream().map(Object::toString).collect(Collectors.joining("\n","actual:\n","\n"))
                +  expected.stream().map(Object::toString).collect(Collectors.joining("\n","expected:\n","\n")));
        for (int i = 0; i < actual.size(); i++) {

            assertion.accept(actual.get(i), expected.get(i));
        }
    }

    //==================================================================================================================
    // Tests
    //==================================================================================================================

    @Test
    public void GATK3LargeConcoranceTest() throws IOException {
        assertVariantContextsMatch(getTestFile("HCOutput.NoAnnotations.vcf"), new File(getToolTestDataDir() + "expected/integrationTest.vcf"), Arrays.asList("-G", "Standard", "-G", "AS_Standard", "-L", "20:10000000-10100000", "-I", NA12878_20_21_WGS_bam), b37_reference_20_21);
    }

    //TODO these tests must be modernized :-(
    @Test
    public void testHasAnnotsNotAsking1() throws IOException {
        final File expected = new File(getToolTestDataDir() + "expected/testHsAnnotsNotAsking1.vcf");
        final VCFHeader header = getHeaderFromFile(expected);
        runVariantAnnotatorAndAssertSomething(getTestFile("vcfexamplemultisample.vcf"), new File(getToolTestDataDir() + "expected/testHsAnnotsNotAsking1.vcf"), Arrays.asList( "-I", largeFileTestDir + "CEUTrio.multisample.b37.1M-1M50k.bam"),
                (a, e) -> {
                    // We need to filter out sites where we saw a DP of 250 because we are comparing the results to GATK3, which downsamples to 250 reads per sample, which GATK4 does not currently support.
                    if (!e.getGenotypes().stream().anyMatch(g -> g.hasDP() && g.getDP() >= 250)) {
                        VariantContextTestUtils.assertVariantContextsAreEqualAlleleOrderIndependent(a, e, ATTRIBUTES_TO_IGNORE, header);
                    }
                },
                b37_reference_20_21);
    }

    @Test
    public void testHasAnnotsAsking1() throws IOException {
        final File expected = new File(getToolTestDataDir() + "expected/testHasAnnotsAsking1.vcf");
        final VCFHeader header = getHeaderFromFile(expected);
        runVariantAnnotatorAndAssertSomething(getTestFile("vcfexamplemultisample.vcf"), new File(getToolTestDataDir() + "expected/testHasAnnotsAsking1.vcf"), Arrays.asList("-G", "Standard", "-I", largeFileTestDir + "CEUTrio.multisample.b37.1M-1M50k.bam"),
                (a, e) -> {
                    // We need to filter out sites where we saw a DP of 250 because we are comparing the results to GATK3, which downsamples to 250 reads per sample, which GATK4 does not currently support.
                    if (!e.getGenotypes().stream().anyMatch(g -> g.hasDP() && g.getDP() >= 250)) {
                        VariantContextTestUtils.assertVariantContextsAreEqualAlleleOrderIndependent(a, e, ATTRIBUTES_TO_IGNORE, header);
                    }
                },
                b37_reference_20_21);
    }

    @Test
    public void testNoAnnotsNotAsking1() throws IOException {
        final File expected = new File(getToolTestDataDir() + "expected/testHsAnnotsNotAsking1.vcf");
        final VCFHeader header = getHeaderFromFile(expected);
        runVariantAnnotatorAndAssertSomething(getTestFile("vcfexamplemultisampleempty.vcf"), new File(getToolTestDataDir() + "expected/testHasNoAnnotsNotAsking1.vcf"), Arrays.asList( "-I", largeFileTestDir + "CEUTrio.multisample.b37.1M-1M50k.bam"),
                (a, e) -> {
                    // We need to filter out sites where we saw a DP of 250 because we are comparing the results to GATK3, which downsamples to 250 reads per sample, which GATK4 does not currently support.
                    if (!e.getGenotypes().stream().anyMatch(g -> g.hasDP() && g.getDP() >= 250)) {
                        VariantContextTestUtils.assertVariantContextsAreEqualAlleleOrderIndependent(a, e, ATTRIBUTES_TO_IGNORE, header);
                    }
                },
                b37_reference_20_21);
    }


    @Test
    public void testNoAnnotsAsking1() throws IOException {
        final File expected = new File(getToolTestDataDir() + "expected/testHasNoAnnotsAsking1.vcf");
        final VCFHeader header = getHeaderFromFile(expected);
        runVariantAnnotatorAndAssertSomething(getTestFile("vcfexamplemultisampleempty.vcf"), new File(getToolTestDataDir() + "expected/testHasNoAnnotsAsking1.vcf"), Arrays.asList("-G", "Standard", "-I", largeFileTestDir + "CEUTrio.multisample.b37.1M-1M50k.bam"),
                (a, e) -> {
                    // We need to filter out sites where we saw a DP of 250 because we are comparing the results to GATK3, which downsamples to 250 reads per sample, which GATK4 does not currently support.
                    if (!e.getGenotypes().stream().anyMatch(g -> g.hasDP() && g.getDP() >= 250)) {
                        VariantContextTestUtils.assertVariantContextsAreEqualAlleleOrderIndependent(a, e, ATTRIBUTES_TO_IGNORE, header);
                    }
                },
                b37_reference_20_21);
    }
    @Test
    public void testOverwritingHeader() throws IOException {
        IntegrationTestSpec spec = new IntegrationTestSpec(
                baseTestString() + STANDARD_ANNOTATIONS + "--variant " + getToolTestDataDir() + "vcfexample4.vcf  -L 20:10,001,292",
                Arrays.asList(getToolTestDataDir() + "expected/testReplaceHeader.vcf"));

        spec.executeTest("test overwriting header", this);
    }

    @Test
    public void testNoReads() throws IOException {
        IntegrationTestSpec spec = new IntegrationTestSpec(
                baseTestString() + STANDARD_ANNOTATIONS + "--variant " + getToolTestDataDir() + "vcfexample3empty.vcf -L " + getToolTestDataDir() + "vcfexample3empty.vcf",
                Arrays.asList(getToolTestDataDir() + "expected/" + "testNoReads.vcf"));
        spec.executeTest("not passing it any reads", this);
    }

    @Test
    public void testMultipleIdsWithDbsnp() throws IOException {
        IntegrationTestSpec spec = new IntegrationTestSpec(
                " -O %s --alwaysAppendDbsnpId --dbsnp " + dbsnp_138_b37_20_21_vcf + STANDARD_ANNOTATIONS + "--variant " + getToolTestDataDir() + "vcfdbsnpwithIDs.vcf -L " + getToolTestDataDir() + "vcfdbsnpwithIDs.vcf",
                Arrays.asList(getToolTestDataDir() + "expected/testMultipleIdsWithDbsnp.vcf"));
        spec.executeTest("adding multiple IDs with dbSNP", this);
    }

    @Test
    public void testDBTagWithHapMap() throws IOException {
        IntegrationTestSpec spec = new IntegrationTestSpec(
                baseTestString() + " --comp H3:" + getToolTestDataDir() + "fakeHM3.vcf" + STANDARD_ANNOTATIONS + "--variant " + getToolTestDataDir() + "vcfexample3empty.vcf -L " + getToolTestDataDir() + "vcfexample3empty.vcf",
                Arrays.asList(getToolTestDataDir() + "expected/testDBTagWithHapMap.vcf"));
        spec.executeTest("getting DB tag with HM3", this);
    }

    @Test
    public void testDBTagWithTwoComps() throws IOException {
        IntegrationTestSpec spec = new IntegrationTestSpec(
                baseTestString() + " --comp H3:" + getToolTestDataDir() + "fakeHM3.vcf --comp foo:" + getToolTestDataDir() + "fakeHM3.vcf " + STANDARD_ANNOTATIONS + " --variant " + getToolTestDataDir() + "vcfexample3empty.vcf -L " + getToolTestDataDir() + "vcfexample3empty.vcf",
                Arrays.asList(getToolTestDataDir() + "expected/testDBTagWithTwoComps.vcf"));
        spec.executeTest("getting DB tag with 2 comps", this);
    }

    @Test
    public void testNoQuals() throws IOException {
        // NOTE, this test is asserting that the QD calculation is dependant on existing QUAL field, the values themselves are subject to random jitter
        Utils.resetRandomGenerator();
        IntegrationTestSpec spec = new IntegrationTestSpec(
                baseTestString() + " --variant " + getToolTestDataDir() + "noQual.vcf -I "+NA12878_20_21_WGS_bam+" -L " + getToolTestDataDir() + "noQual.vcf -A QualByDepth",
                Arrays.asList(getToolTestDataDir() + "expected/noQual.vcf"));
        spec.executeTest("test file doesn't have QUALs", this);
    }

    @Test
    public void testUsingExpression() throws IOException {
        assertVariantContextsMatch(getTestFile("vcfexample3empty.vcf"),
                new File(getToolTestDataDir() + "expected/testUsingExpression.vcf"),
                Arrays.asList("--resourceAlleleConcordance",  "--resource",  "foo:" + getToolTestDataDir() + "targetAnnotations.vcf",
                        "-G", "Standard", "-E", "foo.AF", "-L", getToolTestDataDir()+"vcfexample3empty.vcf"), b37_reference_20_21);
    }

    @Test
    public void testUsingExpressionAlleleMisMatch() throws IOException {
        assertVariantContextsMatch(getTestFile("vcfexample3empty-mod.vcf"),
                new File(getToolTestDataDir() + "expected/testUsingExpressionAlleleMisMatch.vcf"),
                Arrays.asList("--resourceAlleleConcordance",  "--resource",  "foo:" + getToolTestDataDir() + "targetAnnotations.vcf",
                        "-G", "Standard", "-E", "foo.AF", "-L", getToolTestDataDir()+"vcfexample3empty-mod.vcf"), b37_reference_20_21);
    }

    @Test
    public void testUsingExpressionMultiAllele() throws IOException {
        IntegrationTestSpec spec = new IntegrationTestSpec(
                baseTestString() + " --resource foo:" + getToolTestDataDir() + "targetAnnotations-multiAllele.vcf" + STANDARD_ANNOTATIONS + "--variant " + getToolTestDataDir() + "vcfexample3empty-multiAllele.vcf -E foo.AF -E foo.AC -L " + getToolTestDataDir() + "vcfexample3empty-multiAllele.vcf",
                Arrays.asList(getToolTestDataDir() + "expected/testUsingExpressionMultiAllele.vcf"));

        spec.executeTest("ExpressionTestt--" + testFile, this);
    }

    @Test
    public void testFilterInExpression() throws IOException {
        /* The order of filters in the output seems platform-dependent. May need to change htsjdk to make the order consistent across platforms. [Sato] */
        IntegrationTestSpec spec = new IntegrationTestSpec(
                baseTestString() + " --resource foo:" + getToolTestDataDir() + "annotationResourceWithFilter.vcf" + STANDARD_ANNOTATIONS + "--variant " + getToolTestDataDir() + "vcfexample3empty-multiAllele.vcf -E foo.FILTER -L " + getToolTestDataDir() + "vcfexample3empty-multiAllele.vcf",
                Arrays.asList(getToolTestDataDir() + "expected/testFilterInExpression.vcf"));

        spec.executeTest("ExpressionTestt--" + testFile, this);
    }

    @Test
    public void testUsingExpressionWithID() throws IOException {
        IntegrationTestSpec spec = new IntegrationTestSpec(
                baseTestString() + " --resource foo:" + getToolTestDataDir() + "targetAnnotations.vcf" + STANDARD_ANNOTATIONS + "--variant " + getToolTestDataDir() + "vcfexample3empty.vcf -E foo.ID -L " + getToolTestDataDir() + "vcfexample3empty.vcf",
                Arrays.asList(getToolTestDataDir() + "expected/testUsingExpressionWithID.vcf"));

        spec.executeTest("ExpressionTestt--" + testFile, this);
    }

//
//
//    @Test(enabled = true)
//    public void testChromosomeCountsPed() {
//        final String       nD5 = "0a18fe81dde8d0f94f9ac5e5f65d00d5";
//        WalkerTestSpec spec = new WalkerTestSpec(
//                "-T VariantAnnotator -R " + b37KGReference + " -A ChromosomeCounts --variant:vcf " + privateTestDir + "ug.random50000.subset300bp.chr1.family.vcf" +
//                        " -L " + privateTestDir + "ug.random50000.subset300bp.chr1.family.vcf --no_cmdline_in_header -ped " + privateTestDir + "ug.random50000.family.ped -o %s", 1,
//                Arrays.asList(MD5));
//        executeTest("Testing ChromosomeCounts annotation with PED file", spec);
//    }
//
    //TODO this has no arguments just yet
//    @Test(enabled = true)
//    public void testInbreedingCoeffPed() {
//        final String MD5 = "95408408863cc81c63ec3c53716bf9d2";
//        WalkerTestSpec spec = new WalkerTestSpec(
//                "-T VariantAnnotator -R " + b37KGReference + " -A InbreedingCoeff --variant:vcf " + privateTestDir + "ug.random50000.subset300bp.chr1.family.vcf" +
//                        " -L " + privateTestDir + "ug.random50000.subset300bp.chr1.family.vcf --no_cmdline_in_header -ped " + privateTestDir + "ug.random50000.family.ped -o %s", 1,
//                Arrays.asList(MD5));
//        executeTest("Testing InbreedingCoeff annotation with PED file", spec);
//    }

    @Test
    public void testAlleleTrimming() throws IOException {
        // This test makes sure that the expression code works in a complex case with many overlapping variant contexts
        IntegrationTestSpec spec = new IntegrationTestSpec(
                "-O %s -A InbreedingCoeff --variant " + getToolTestDataDir() + "alleleTrim.vcf" +
                        " --resource exac:" + getToolTestDataDir() +  "exacAlleleTrim.vcf  -E exac.AC_Adj",
                Arrays.asList(getToolTestDataDir() + "expected/testAlleleTrimming.vcf"));
        spec.executeTest("Testing allele trimming annotation", this);
    }

    @Test
    public void testStrandBiasBySample() throws IOException {
        // Created variants via HalotypeCaller GATK3 with no default annotations
        final File outputVCF = getTestFile("HCOutputNoAnnotations.vcf");

        // Created variant via HalotypeCaller GATK3; include StrandBiasBySample, exclude FisherStrand annotation
        //             re-Annotate the variant with VariantAnnotator using FisherStrand annotation
        final File outputVCFNoFS = getTestFile("HCOutputNoFSAnnotation.vcf");

        final File outputWithAddedFS = createTempFile("variantannotator", ".vcf");

        final ArgumentsBuilder args = new ArgumentsBuilder();
        args.addReference(new File(b37_reference_20_21));
        args.addVCF(outputVCFNoFS);
        args.addOutput(outputWithAddedFS);
        args.add("-L 20:10130000-10134800");
        args.add("-A FisherStrand");
        runCommandLine(args);

        // confirm that the FisherStrand values are identical for the two pipelines
        final VCFCodec codec = new VCFCodec();
        final FileInputStream s = new FileInputStream(outputVCF);
        final LineIterator lineIterator = codec.makeSourceFromStream(new PositionalBufferedStream(s));
        codec.readHeader(lineIterator);

        final VCFCodec codecAnn = new VCFCodec();
        final FileInputStream sAnn = new FileInputStream(outputWithAddedFS);
        final LineIterator lineIteratorAnn = codecAnn.makeSourceFromStream(new PositionalBufferedStream(sAnn));
        codecAnn.readHeader(lineIteratorAnn);

        while( lineIterator.hasNext() && lineIteratorAnn.hasNext() ) {
            final String line = lineIterator.next();
            Assert.assertFalse(line == null);
            final VariantContext vc = codec.decode(line);

            final String lineAnn = lineIteratorAnn.next();
            Assert.assertFalse(lineAnn == null);
            final VariantContext vcAnn = codecAnn.decode(lineAnn);

            Assert.assertTrue(vc.hasAttribute("FS"));
            Assert.assertTrue(vcAnn.hasAttribute("FS"));
            Assert.assertEquals(vc.getAttributeAsDouble("FS", 0.0), vcAnn.getAttributeAsDouble("FS", -1.0));
        }

        Assert.assertFalse(lineIterator.hasNext());
        Assert.assertFalse(lineIteratorAnn.hasNext());
    }

}
