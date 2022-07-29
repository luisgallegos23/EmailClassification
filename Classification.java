import java.io.File;
import java.io.FileNotFoundException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;

public class Classification {

    private static HashMap<String, Integer> SpamWords = new HashMap<>(); //contains the spam vocab w/ times used
    private static HashMap<String, Integer> HamWords = new HashMap<>(); //contains the ham vocab w/ tiems used
    private static double TotalSpam; //total training emails used for spam vocab
    private static double TotalHam; //total training emails used for ham vocab
    private static int HamContaining; //number of ham words used in test email
    private static int SpamContaining; //number of spam words used in test email
    private static int TotalTest; //number of total email read during testing
    private static int TotalRight; //number of total emails classified correctly

    public static void main(String[] args) throws FileNotFoundException {
        trainFiles(getTrainSpamFile(),true);
        trainFiles(getTrainHamFile(),false);
        CompleteVocab();
        String testspam = getTestSpamFile();
        String testham = getTestHamFile();
        testFiles(testspam,true);
        testFiles(testham,false);
        System.out.printf("Total: %s/%s emails classified correctly.",TotalRight,TotalTest);
        
    }

    /** Takes the Training file from the user for the Spam vocabulary */
    public static String getTrainSpamFile(){
        Scanner scan = new Scanner(System.in);
        System.out.print("What is your spam training file?");
        String  file = scan.nextLine();
        scan.close();
        return file;
    }

    /** Takes the Training file from the user for the Ham vocabulary */
    public static String getTrainHamFile(){
        Scanner scan = new Scanner(System.in);
        System.out.print("What is your ham training file?");
        String file = scan.nextLine();
        scan.close();
        return file;
    }

    /** Takes the Training file from the user for the Spam vocabulary */
    public static String getTestSpamFile(){
        Scanner scan = new Scanner(System.in);
        System.out.print("What is your spam testing file?");
        String file = scan.nextLine();
        scan.close();
        return file;
    }

    /** Takes the Training file from the user for the Ham vocabulary */
    public static String getTestHamFile(){
        Scanner scan = new Scanner(System.in);
        System.out.print("What is your ham testing file?");
        String file = scan.nextLine();
        scan.close();
        return file;
    }

    /** Reads the training files based on the input from the users
     * boolean type = true if the file is meant for Spam classification
     * boolean type = false if the file is meant for Ham classification **/
    public static void trainFiles(String document, boolean type) throws FileNotFoundException {
        HashMap<String, Integer> map = new HashMap<>();
        File file = new File(document);
        Scanner doc = new Scanner(file);
        ArrayList<String> wordsread= new ArrayList<>();
        double numofspam = 0.0;
        while(doc.hasNextLine()){
            String line = doc.nextLine().toLowerCase();
            String[] words  = line.split(" ");
            for(String word: words){
                if(word.equals("</subject>") || word.equals("<body>") || word.equals("<subject>")){
                    continue;
                }
                if(word.equals("</body>")){
                    numofspam++;
                    wordsread.clear();
                    continue;
                }
                if(word.equals("")){
                    continue;
                }
                //Conditional statements meant to only store words once for each email
                if(!map.containsKey(word)){
                    map.put(word,1);
                    wordsread.add(word);
                }else if (map.containsKey(word) && !wordsread.contains(word)){
                    int x = map.get(word);
                    int y = x+1;
                    map.replace(word,x, y);
                    wordsread.add(word);
                }
            }
        }
        //Depending on the type, the vocab will either be stored in a Hashmap for Spam words
        // or a Hashmap for Ham words
        if(type){
            TotalSpam = numofspam;
            SpamWords = map;
        }else{
            TotalHam = numofspam;
            HamWords = map;
        }
        doc.close();
    }

    public static void testFiles(String document,boolean type) throws FileNotFoundException {
        ArrayList<String> emailvocab = new ArrayList<>();
        File file = new File(document);
        Scanner doc = new Scanner(file);
        int numemail = 1;
        while(doc.hasNextLine()){
            String line = doc.nextLine().toLowerCase();
            String[] words  = line.split(" ");
            for(String word: words){
                if(word.equals("</subject>") || word.equals("<body>") || word.equals("<subject>")){
                    continue;
                }
                if(word.equals("</body>")){
                    TotalTest++;
                    double ham = calculateHam(emailvocab);
                    double spam = calculateSpam(emailvocab);
                    classifyEmail(numemail,ham,spam,type);
                    emailvocab.clear();
                    numemail++;
                    HamContaining = 0;
                    SpamContaining = 0;
                    continue;
                }
                if(word.equals("")){
                    continue;
                }
                emailvocab.add(word);
                }
            }
            doc.close();
        }


    /** Calculate the Map Hypothesis of Ham
     * Method also smooths the prop of each feature **/
    public static double calculateHam(ArrayList<String> emailvocab){
        double x = TotalHam/(TotalHam+TotalSpam);
        double hamprob = Math.log(x);
        for(Map.Entry<String,Integer> word: HamWords.entrySet()){
            if(emailvocab.contains(word.getKey())){
                double num = word.getValue()+1; //+1 smooths
                double den = TotalHam + 2; //+2 smooths
                double value = num/den;
                hamprob += Math.log(value);
                HamContaining++;
            }if(!emailvocab.contains(word.getKey())){
                double num = word.getValue()+1;
                double den = TotalHam + 2;
              double value = 1-(num/den);
              hamprob += Math.log(value);
            }
        }
        BigDecimal v = new BigDecimal(hamprob);
        v = v.setScale(3, RoundingMode.HALF_UP);//rounds to three decimals
        return v.doubleValue();
    }

    /** Calculate the Map Hypothesis of Spam
     * Method also smooths the prop of each feature **/
    public static double calculateSpam(ArrayList<String> emailvocab){
        double x = TotalSpam/(TotalHam+TotalSpam);
        double spamprob = Math.log(x);
        for(Map.Entry<String,Integer> word: SpamWords.entrySet()){
            if(emailvocab.contains(word.getKey())){
                double num = word.getValue()+1;
                double den = TotalSpam + 2;
                double value = num/den;
                spamprob += Math.log(value);
                SpamContaining++;
            }if(!emailvocab.contains(word.getKey())){
                double num = word.getValue()+1;
                double den = TotalSpam + 2;
                double value = 1-(num/den);
                spamprob += Math.log(value);
            }
        }
        BigDecimal v = new BigDecimal(spamprob);
        v = v.setScale(3, RoundingMode.HALF_UP);
        return v.doubleValue();
    }

    /** Classifies the email to either be spam or ham**/
    public static void classifyEmail(int num, double ham, double spam, boolean type){
        String classemail;
        boolean match;
        //Conditions give weather the email was classified as ham=false and spam=true
        if(ham > spam){
            match = false;
            classemail = "ham";
        }else{
            match = true;
            classemail = "spam";
        }
        // Compares the type of file read to the type the email was classified
        // Determines if the email was classified correct or not, if classemail and type equal, then email
        // was classified correctly
        String correct;
        if(match == type){
            TotalRight++;
            correct = "right";
        }else{
            correct = "wrong";
        }
        //Format the output based on the test file type(spam or ham)
        if(type){
            System.out.printf("TEST %s %s/%s features true %s %s %s %s%n",num,SpamContaining,SpamWords.size(),spam,ham,classemail,correct);
        }else{
            System.out.printf("TEST %s %s/%s features true %s %s %s %s%n",num,HamContaining,HamWords.size(),spam,ham,classemail,correct);
        }
    }

    /** Complete the two hashmaps with words that are in ham and not in spam
     * set their value to zero, and vis versa */
    public static void CompleteVocab(){
        for(Map.Entry<String,Integer> word: SpamWords.entrySet()){
            if(!HamWords.containsKey(word.getKey())){
                HamWords.put(word.getKey(),0);
            }
        }
        for(Map.Entry<String,Integer> word: HamWords.entrySet()){
            if(!SpamWords.containsKey(word.getKey())){
                SpamWords.put(word.getKey(),0);
            }
        }
    }
}
