package org.broadinstitute.hellbender.tools.spark.sv.discovery.prototype;

import htsjdk.samtools.SAMRecord;
import org.broadinstitute.hellbender.GATKBaseTest;
import org.broadinstitute.hellbender.tools.spark.sv.discovery.AlignedContig;
import org.broadinstitute.hellbender.utils.Utils;
import org.broadinstitute.hellbender.utils.read.ArtificialReadUtils;
import org.broadinstitute.hellbender.utils.read.GATKRead;
import org.junit.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.broadinstitute.hellbender.tools.spark.sv.StructuralVariationDiscoveryArgumentCollection.DiscoverVariantsFromContigsAlignmentsSparkArgumentCollection.GAPPED_ALIGNMENT_BREAK_DEFAULT_SENSITIVITY;
import static org.broadinstitute.hellbender.tools.spark.sv.discovery.DiscoverVariantsFromContigAlignmentsSAMSpark.SAMFormattedContigAlignmentParser.parseReadsAndOptionallySplitGappedAlignments;

public class AssemblyContigWithFineTunedAlignmentsUnitTest extends GATKBaseTest {

    @Test(groups = "sv", dataProvider = "createData")
    void test(final AlignedContig alignedContig, final boolean expectedHasIncompletePicture) {

        Assert.assertEquals(new AssemblyContigWithFineTunedAlignments(alignedContig).hasIncomePicture(), expectedHasIncompletePicture);


    }

    @DataProvider(name = "createData")
    private Object[][] createTestData() {
        final List<Object[]> data = new ArrayList<>(20);

        final List<String> bases1 = Arrays.asList("ACCAAGGGCAAAATTGTTCCACAGCATGAAAGAAATCCATAAGTTTTTCTGTATCAACTTTTACCCTACCATGCTTCAAGAGCTGCTGTAGCAAGCTCAAATACATGATGTACTTACTTTCAGTTTGTCCCATTTGTGTCCCTAGCTTTCTCCGAGTGCCCCGCTTACCTGCAGAGCTTGAAACTTTTTCATCCTTGGGAGTCCTTTGTCTGTTGGTCCTCTGTTTCACACACTTGAGTGTTCCTTCACCGGATTCTTTCAGGCCCCACGTTGGGCGCCAGAATGTTGGGGACCAGCCTCAACACCACCCGTAGGGTACCCAAAGTCCAATGGTGACAAAGGAATGAGAAAAGACAGGTTAAGAGTTCATAAAGGTGGGAGCCAGGGGACCAGTTGCAAAATGGAGGCTGCAAAAGGCTCAGAGCTCTGGTCTCCACACTATTTATTGGGTACAATCACTTAGATGTAAAAAGCAGATGTTCAGGGTGAAACAGTGAAAGGGTGGCAGTGCATCATAGGTGTAATTTATAGCAATAGTAGTTTAAATGAATCTCCTTGTGCTCAGTGTATCTTTAACTTATTGGAGAGTAGCTAGTGGGAGTGGGCTTAACTAGGAGCCTGCATGTCTGTCCGCATTCCCATGCTTCAAAGGAGTGTCTTTCTCCTGGAACACAGTGTTTACAAATAAGAGAGCGGGTCTCGCTCTGAGCATGGGAACATGATGGCAATTAGGAGGCTTTCCTCCTCAGAGGCCTTTTGTGGCTTTTCACAACTTATTTTCCTATATTTTTATGGCCAGTTTATACAGGCACCCCACAAGCCCTTTTCCCAACAATTTAAGTTTCCTGAGGCCTCCTGAGAAGCAGAAGCCACTATACTTCTTGTACAGCCTATAGAACCTCTTTTCTTTATAAATTACTCAGTTTCAGGTATTCCTTTATAACAACGCGAGAATGGACTAATAAAATCACCACATCATCATCCTCATCACCATTATCATCACCATCAACATCACTGTCATCATCACCATCATCATTATCACCATCACCATCATTATAGTTACCATCACCATAATTATCACCATCAACATCACCATCGGTGCCATCAACATCATCATCATCACCATCACCATCATCATCATCATTACCATTACCATCACCATCATCATCTTCACCATCACCATTGCACCATCACCATTATCATCATCATAATCACCATCAACATCACCATAATAATCATCACCATTATCATCATTATCACCATCACCATCATCATCATCATTACCATTACCATCACCATCATCATCTTCACCATCACCATCACCATTGCACCATCACCATTATCATCATCTTAATCACTATCAACATCACCATAATAATCACCATCACCATCACCATTATCATCATTATCACCATCACCATTATCACCATCATAATCATCATGAAATTACCATAATCATACTCACCATTACTATCACTGTCATCACCATCACAATCATTACCACTAACACCACCACCATCATCATCATCATCACCACCACCATCATCATCATTATCCTAACAATAAAGATGGCAGAACAAATGAATCTCATTTGTTAATGCCCCACATGGTGGCACTGTGCTGAGGGCCCTTATCTGCTACATCTCATCACTCAGCCTTATGTTGCTTCCCTCAGACCTTTAAGGATTTCTAAGCAAAATGGGATAACCTTATTCCTGGAAGAATTAATTGCTTCTTGAGTCAATAAACTATATTGAGGACCTAGTATGTGGTGGGCATCCATCAGAGAGCAAAACCAGGTGTGGTTCCTGCCCTCGTGGAGCTTACAGTCCAGTGGGGAGACAGATATTACTCATTACATACCGAATGGACACTTACAGATAGAGGTAAGTAGCTTGATAGAAAGTTCCATGGGGTTGGCCAGGTGTGGTGGCTCATGCCTGTAATCCCAGCATTTTGGGAGGCTGAGGTGGGTGGATCACAAGGTCAGGAGTTTGAGACCAGCCTGGCCAATATGGTGAAACCCTGTCTCTACTAAAAATACAAAAATTACCTGGGTGTGGTGGTGCGGGCCTGTAGTCCCAGTTACTTGGGAGGCTGAGGCAGAAGAATCGCTTAAACCCGGGAGGCGGAGGTTGCAGTTAGCCAAGATTGCACCACTGCACTCCAGCCTGGGTGACAGAGTGAGACTCCATCTCAA",
                "ATCACCATCACCATTATCATCATTATCACCATCACCATTATCACCATCATAATCATCATGAAATTACCATAATCATACTCACCATTACTATCACTGTCATCACCATCACAATCATTACCACTAACACCACCACCATCATCATCATCATCACCACCACCATCATCATCATTATCCTAACAATAAAGATGGCAGAACAAATGAATCTCATTTGTTAATGCCCCACATGGTGGCACTGTGCTGAGGGCCCTTATCTGCTACATCTCATCACTCAGCCTTATGTTGCTTCCCTCAGACCTTTAAGGATTTCTAAGCAAAATGGGATAACCTTATTCCTGGAAGAATTAATTGCTTCTTGAGTCAATAAACTATATTGAGGACCTAGTATGTGGTGGGCATCCATCAGAGAGCAAAACCAGGTGTGGTTCCTGCCCTCGTGGAGCTTACAGTCCAGTGGGGAGACAGATATTACTCATTACATACCGAATGGACACTTACAGATAGAGGTAAGTAGCTTGATAGAAAGTTCCATGGGGTTGGCCAGGTGTGGTGGCTCATGCCTGTAATCCCAGCATTTTGGGAGGCTGAGGTGGGTGGATCACAAGGTCAGGAGTTTGAGACCAGCCTGGCCAATATGGTGAAACCCTGTCTCTACTAAAAATACAAAAATTACCTGGGTGTGGTGGTGCGGGCCTGTAGTCCCAGTTACTTGGGAGGCTGAGGCAGAAGAATCGCTTAAACCCGGGAGGCGGAGGTTGCAGTTAGCCAAGATTGCACCACTGCACTCCAGCCTGGGTGACAGAGTGAGACTCCATCTCAA",
                "CCATCAACATCATCATCATCACCATCACCATCATCATCATCATTACCATTACCATCACCATCATCATC");
        final List<String> cigars1 = Arrays.asList("1017M3I52M1128S", "1383H817M", "1101H68M1031H");
        final List<String> chromosomes1 = Arrays.asList("chr1", "chr1", "chr7");
        final List<Integer> positions1 = Arrays.asList(19931877, 19932932, 355947);
        final List<Boolean> reverseStrands1 = Arrays.asList(false, false, false);
        final List<Boolean> suppStatus1 = Arrays.asList(false, true, true);

        final List<SAMRecord> reads1 = createSAMRecordsWithEssentialInfo(bases1, cigars1, chromosomes1, positions1, reverseStrands1, suppStatus1);

        final AlignedContig alignedContig1 = parseReadsAndOptionallySplitGappedAlignments(reads1, GAPPED_ALIGNMENT_BREAK_DEFAULT_SENSITIVITY, true);

        data.add(new Object[]{alignedContig1, false});



        final List<String> bases2 = Arrays.asList("CTGGAGTGCAATGGCATGATATTGGCTCACTGCAACCTCCACCTCCTGGGTTCAAGCAATTCTCCTGCCTCAGCTTCTAGAGTAGCTGGGATTACAGGTGCACACTACCACGCCCAGCTAATTTTTGTATTTTTATTAGAGATGGGGTTTCATCATGTTGGTCAGGCTGGTCTCGAACTCCTGACCTCAGGTGATTGTCCTGCCTCAGCATCCCAAAGTGCTGGGATTACAGGCATGAGGCACCGCGCCCAGCCAGCATGGAGGTATTTGAGAGCAACAGTGATCAGAACCATTTGGTTCAAGCAGCGGTTTTAAAACGGAAGTGGAGAAGGAATTAGCAGATCCCTGACATCCTCTTCAATCAGAGTTCCTCCATTGTGAACTGGTTTACATGTCAGCATTATGGATTTTGGTGCAACACCTGCCCCCAACAGGAAGAAAAGAAGAAAAAGAAAGAAGAGGAAGGAAGAAGAGAAAGACAAAGAAGAAGAAGGAGGAGGAGGCGGCAGGAGGAGTAGGAGGGAGGAAGAAGGAGGAGGAAGAAAAAGAGGAAGAAGAAAGGAGGAAGGAAGAAGAAAGAAGAATTGGGAGGCTGAGACAGGCAGATCACAAGGTCAGGAGTTCAAGACCAGCCTGGCCAACATGGTGAAACCCCATCACTTCTAAAAATACAAAAATTAGCTGGGCGTGTTGGCACATGCCTATAATCCCAGCTACTTGGGAGGCTGAGGCAGGAGAATTGCTTGAACCTGGGAGGCGGAGGTTGCAGTGAGCAGAGAGCTCGCCACTGCACTCCAGTCTGGGCAATGAGCGAGACTGTCTTGAAAAAAAAAAAAGGAAGGAAAGAAGGAGGAGGAGGAGAAGAAAGAAGAAGTTTTATTATTGTTATTTTTTGAGATGGAGTCTTGCTCTGTAGCCCAGGCTGGAATACAGTGGCACACTCTTGACTCCCTGCCACCTCTGACTGCTGGGTTCAAAGGAGAAGGCGGAAGAAGAAGGAAGAAGAAGAAAGAAGGAGAAAGGCTGGGTGCAGTGGCTCACACCTGTAATCTCAGCACTTTGGGAGGCCGAGGCAGGTGAATCACAAGGTAAGGAGTTCGAGACCAGCCTGGCCAACTGGTGAAACCCTGCCTCTACTAAAAGTACAAAAATTAGCCGGGCGTGGTCTGAGGCAGGAGAATCGCTTTAACCTGGGAGGAGGAGGTTGCAGTCAGTCAAGATGGCGCCACTGCACTCCAGCCTGGGTGACAGAGTGAGACTTTGTCTCAAAAAAAAAAAAGGAAAGAAGGAGAAGAAAGAAGGAGGAGGAGGAGAAGAAGAAGAGGAAAAGGAGGGGGAGGAGAAGGGGAGGGGGAGGAAAGAGGAGGAGAAGAAAGAAGAAGGAGAAGGAAGAAAGGAGGGAGGAGGAGGAGGAGAAGGGGGAGGGGGAGGAAGGAGGAGGAGAAGAATGAGGAGAAGGAGAAGAAGAAGGAAGAAGAAAGAAGGAAGAAGAAGGAGGAGGAGGAGGGGGAGGAGGGGGAGGGGGAGGAGGAGGAGGAGAGAAGGAGGAGAAAAGTAGTTGAGGCCCAAACACCAAGAGGGAGCAAAGATTGAAAAGATGAGATGAGCCATGAAAGCAAGTACAGGAGTTACTGATGGTACTGGGGAGCCCGTGTAGGTTTAGGATCTGAGCTTTTGGAAGATTGATTGGGTAGCCTTTGAGCCACCTGATAAGTGGAAAGAACAAGAGAGGCTGGATCTGTGTTTTCAGGAAGCATATGTTGGCCCAGCAATTGCTGGCTTGTAGTGAGGAGGCACACAACTGGCCTAGGACAGTGGTCATGAAAATGCAGAGGAGGTAAAGTCCCTGCACTCCTAGGGAGACTAGTCCTGATGTCAGTCTGGAGTCAGTCAGAATGGTGTCCTCTCCCTCCCTGCACTACCCAGCCCAGTCAGTGGGAGGACTTCCTCAATTCCAGTAGCCATTCAAGTCCCTGGAATTGGTGGCTGTCACTTGCAAACTATAGCCACTTGAGCAGAAATGGGCCAGGATTACTTATCTTTAATCTGCATATCATTGGGAGGCACTTACCTGCTAGCTCTGGCTAAAAACTAGAGCAACCCTGGCCTGCCGTAGCTCCTGCTGCCCAGACAACTCCTCCAATATGAAAGGGATGAGGGGAACTCAAAGTTACAATGTCCTACTTGGAGCAGTAAGTTCAGTAGACATATCACTTGCCTCATTAACATCAAGCATCCCAAAACCCAGTCTGGGTCAGTTTTGCCCAGAGTGGGGTTTGTAGAACACGGGTTCTCCTGGGATCCTATACCTAGCCCAGAATCAGTTGCAAAAGCCAGGCCATAGCGAATTGTCCTGCCAGCCAGATAGCAGAGAATCTGACGGCAGCAGGCAGAAGGAGCCGCTCCATTGCAGTAAGCCAAGATCGCGCCACTTGCCTCATTACATCAAGCATCCCAAAACCCAGTCTGGGTCAGTTTTGCCCA",
                "GGAGGGGGAGGAGGAGGAGGAGAGAAGGAGGAGAAAAGTAGTTGAGGCCCAAACACCAAGAGGGAGCAAAGATTGAAAAGATGAGATGAGCCATGAAAGCAAGTACAGGAGTTACTGATGGTACTGGGGAGCCCGTGTAGGTTTAGGATCTGAGCTTTTGGAAGATTGATTGGGTAGCCTTTGAGCCACCTGATAAGTGGAAAGAACAAGAGAGGCTGGATCTGTGTTTTCAGGAAGCATATGTTGGCCCAGCAATTGCTGGCTTGTAGTGAGGAGGCACACAACTGGCCTAGGACAGTGGTCATGAAAATGCAGAGGAGGTAAAGTCCCTGCACTCCTAGGGAGACTAGTCCTGATGTCAGTCTGGAGTCAGTCAGAATGGTGTCCTCTCCCTCCCTGCACTACCCAGCCCAGTCAGTGGGAGGACTTCCTCAATTCCAGTAGCCATTCAAGTCCCTGGAATTGGTGGCTGTCACTTGCAAACTATAGCCACTTGAGCAGAAATGGGCCAGGATTACTTATCTTTAATCTGCATATCATTGGGAGGCACTTACCTGCTAGCTCTGGCTAAAAACTAGAGCAACCCTGGCCTGCCGTAGCTCCTGCTGCCCAGACAACTCCTCCAATATGAAAGGGATGAGGGGAACTCAAAGTTACAATGTCCTACTTGGAGCAGTAAGTTCAGTAGACATATCACTTGCCTCATTAACATCAAGCATCCCAAAACCCAGTCTGGGTCAGTTTTGCCCAGAGTGGGGTTTGTAGAACACGGGTTCTCCTGGGATCCTATACCTAGCCCAGAATCAGTTGCAAAAGCCAGGCCATAGCGAATTGTCCTGCCAGCCAGATAGCAGAGAATCTGACGGCAGCAGGCAGAAGGAGCCGCTCCATTGCAGTAAGCCAAGATCGCGCCACTTGCCTCATTACATCAAGCATCCCAAAACCCAGTCTGGGTCAGTTTTGCCCA",
                "CCTCCTCCTCCTCCCCCTCCCCCTCCTCCCCCTCCTCCTCCTCCT");
        final List<String> cigars2 = Arrays.asList("1313M1171S", "1517H967M", "947H45M1492H");
        final List<String> chromosomes2 = Arrays.asList("chr1", "chr1", "chr14");
        final List<Integer> positions2 = Arrays.asList(39043258, 39044558, 70910125);
        final List<Boolean> reverseStrands2 = Arrays.asList(false, false, true);
        final List<Boolean> suppStatus2 = Arrays.asList(false, true, true);

        final List<SAMRecord> reads2 = createSAMRecordsWithEssentialInfo(bases2, cigars2, chromosomes2, positions2, reverseStrands2, suppStatus2);

        final AlignedContig alignedContig2 = parseReadsAndOptionallySplitGappedAlignments(reads2, GAPPED_ALIGNMENT_BREAK_DEFAULT_SENSITIVITY, true);
        data.add(new Object[]{alignedContig2, false});


        final List<String> bases3 = Arrays.asList("CCCGCCTCAGCCTCCCTAAGTGCTGAGATTACAGGCCTGAGCCACTGCGCCAGGCCTGGTTTTTTGGTTTCAAACCACAATAGACATTGCTGGAGAATCAAGCTCATAGTTTCTTTTTACTCTGCATGATATCCCTCCAAAAGCTTGTCTATTCTCATGACTTCATGACAGTTCTTTGCCAATGATTTGCAAACATTATCTCCAGTCTTGATTCTTTCCTAGGCTTTATTTCTAAATACTCACTAGTCATTTCCACTTAGAAGCTTTGTCTTCTTTTCAAACTCAGCATATCCAAAACTGACCTCATCTTGTTGCCCTAACTAGAACATGCCACAACCCATTTCTGCCTATAATGTCATTATTCTCTTAAGGCCCCCATATTTGCAAATCAGGAGTCATCTTTGCCTCCCTTCTCTGAATGCTGCTGTGCTCACAGTCAGTACCATACAGTTAGTACTTGTTCTCAGTTTAATGCACACTTGTTTCATATGACCTCCTAGGGCAGAACCAAGATTCGTGGGTGGAAGTTGCAGGGAGGCAGATTTGCCTCCATATAAGGAAGAACTTTTTAATACTTTGAGCTGTCTGAATGGAATGGACTGCCTCCAGAAGTCTCGGGTTCTCCATCACTGAAGGTGTTTGAGCAGAGGCTGCCTGACTAATTGCTAATTGAAAGGACTGTTCTGGCATAAGATGGTGTTTTTACAAGATGATTTCTAGAATCCCTCTAATCCTGAGAGCCAGTGAGTCGATAGAAGGTAGCTTTGTCTCTCCTGCTAGACTCCCTTAGGACAGGGAGACTATTTTACCTTTCTTTTATATTCTGTACAGCACTTAATTCAGGTGCTGGTCTCTTAATTGCCTAAAGATGATTATTTACAGGTTAATTGATTCTTTTCATTTTGTTCCAATATTTGGTTAAACACCAAATATTGTGGATTTTTTTCCTTTGAAATATCTTCTGTAGTCTGGGCACGGTGGCTCATGCCTGTAATCCCAGCACTTGGCTCACCGCAATTACAGGCATGAGCCACCGTGCCCAGACTACAGAAGATATTTCAAAGGAAATATCTTGGGGAGACCAAGGCAGGTGGATCACCTGAGATCAGGAGTTCGAGACCAGCCAGGCCAACATGGCGAAACCTTGTCTCTACTAAAAATACAAAAATTCGCCGGGCGTGGTGGTGCATGCCTGTAGTTCCAGCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAACCTGGAAGGTGGAGGTTGCAGTGAGCTGAGATCATGCCATCACACTCCAGCCTGGGCAACAGAGTGAGACTCCATCTCAAAAAATTACAGTAATAATAAATAAATACAAATATCTTTTGTAGTAGCATTATTTTAGATGAATAGCAGCTTTTAACCTAACTTTCTAGACCCTAATATCATCACTTGTCTCCACATCCCATTCTGTTTATCAGACAGTACTTGTAAGCTCTTTGCTTTCATCATATTATTCTTTCACTCAAAACCCTTGAATGGCTTCACATTGATCACAACGTCAAATTTAAATTTTCCCTGGCTTTCAGTGCCCTCCATCATCTGGCCCAGCTTGCTCATGCATCCTTGTTCCCTATAATACTCCATGTTCCATGCAGGCTAACCCACTCACATTTTCTGTACATAGCCTGCTTGGTCTCCTTTTTTGCTTTTGACTTGCTCTGGAATGGCTTCCCTTTTTTCTCTTGCCTCTTCAAGATGCCTCACTTCCCTCCTGAAACCAAGCTACCGCCAGTCCTCATTGATTTCCTCTGAACTCTCAGAGCATGTAGTAATTTATGTAATTCAGAACCGTAGCAACAACATTCTGTGAAGAAAAATCTGCAAGAATAGGCTGATAATTTAACTTTCCCTAATCCAACTGGATATTCCCATAATAAAACTTTTAAAAATATAGGCTGGCTGTCATGGTTCACATCTGTAATCCCAGCATTTTGGGAGGCTGAGGCAAGAGGACTGTTTGAGCCCAGGAGTTTGAGACTAGCCTGAGCAACATAGTGAGACTCTGTCTCTATCACACACATA",
                "AAGTGCTGGGATTACAGGCATGAGCCACCGTGCCCAGACTAC",
                "CCCGCCTCAGCCTCCCTAAGTGCTGAGATTACAGGCCTGAGCCACTGCGCCAGGCCTGGTTTTTTGGTTTCAAACCACAATAGACATTGCTGGAGAATCAAGCTCATAGTTTCTTTTTACTCTGCATGATATCCCTCCAAAAGCTTGTCTATTCTCATGACTTCATGACAGTTCTTTGCCAATGATTTGCAAACATTATCTCCAGTCTTGATTCTTTCCTAGGCTTTATTTCTAAATACTCACTAGTCATTTCCACTTAGAAGCTTTGTCTTCTTTTCAAACTCAGCATATCCAAAACTGACCTCATCTTGTTGCCCTAACTAGAACATGCCACAACCCATTTCTGCCTATAATGTCATTATTCTCTTAAGGCCCCCATATTTGCAAATCAGGAGTCATCTTTGCCTCCCTTCTCTGAATGCTGCTGTGCTCACAGTCAGTACCATACAGTTAGTACTTGTTCTCAGTTTAATGCACACTTGTTTCATATGACCTCCTAGGGCAGAACCAAGATTCGTGGGTGGAAGTTGCAGGGAGGCAGATTTGCCTCCATATAAGGAAGAACTTTTTAATACTTTGAGCTGTCTGAATGGAATGGACTGCCTCCAGAAGTCTCGGGTTCTCCATCACTGAAGGTGTTTGAGCAGAGGCTGCCTGACTAATTGCTAATTGAAAGGACTGTTCTGGCATAAGATGGTGTTTTTACAAGATGATTTCTAGAATCCCTCTAATCCTGAGAGCCAGTGAGTCGATAGAAGGTAGCTTTGTCTCTCCTGCTAGACTCCCTTAGGACAGGGAGACTATTTTACCTTTCTTTTATATTCTGTACAGCACTTAATTCAGGTGCTGGTCTCTTAATTGCCTAAAGATGATTATTTACAGGTTAATTGATTCTTTTCATTTTGTTCCAATATTTGGTTAAACACCAAATATTGTGGATTTTTTTCCTTTGAAATATCTT");
        final List<String> cigars3 = Arrays.asList("1064S991M", "1050H42M963H", "961M1094H");
        final List<String> chromosomes3 = Arrays.asList("chr1", "chr16", "chr1");
        final List<Integer> positions3 = Arrays.asList(40050406, 48151, 40049455);
        final List<Boolean> reverseStrands3 = Arrays.asList(true, false, true);
        final List<Boolean> suppStatus3 = Arrays.asList(false, true, true);

        final List<SAMRecord> reads3 = createSAMRecordsWithEssentialInfo(bases3, cigars3, chromosomes3, positions3, reverseStrands3, suppStatus3);

        final AlignedContig alignedContig3 = parseReadsAndOptionallySplitGappedAlignments(reads3, GAPPED_ALIGNMENT_BREAK_DEFAULT_SENSITIVITY, true);
        data.add(new Object[]{alignedContig3, false});

        return data.toArray(new Object[data.size()][]);
    }

    private static List<SAMRecord> createSAMRecordsWithEssentialInfo(final List<String> readBases,
                                                                     final List<String> cigars,
                                                                     final List<String> chromosomes,
                                                                     final List<Integer> positions,
                                                                     final List<Boolean> isReverseStrand,
                                                                     final List<Boolean> isSupplementary) {
        Utils.validateArg(readBases.size() == cigars.size() &&
                        cigars.size() == chromosomes.size() &&
                        chromosomes.size() == positions.size() &&
                        positions.size() == isSupplementary.size() &&
                        isReverseStrand.size() == isSupplementary.size(),
                        "input of different sizes");

        return IntStream.range(0, readBases.size())
                .mapToObj(i -> {
                    final byte[] dummyQuals = new byte[readBases.get(i).length()];
                    Arrays.fill(dummyQuals, (byte)'A');
                    final GATKRead artificialRead = ArtificialReadUtils.createArtificialRead(readBases.get(i).getBytes(), dummyQuals, cigars.get(i));
                    artificialRead.setPosition(chromosomes.get(i), positions.get(i));
                    artificialRead.setIsReverseStrand(isReverseStrand.get(i));
                    artificialRead.setIsSupplementaryAlignment(isSupplementary.get(i));
                    return artificialRead;
                })
                .map(read -> read.convertToSAMRecord(null))
                .collect(Collectors.toList());
    }
}
